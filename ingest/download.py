"""从巨潮资讯网下载上市公司年报 PDF，并维护文档清单 data/manifest.csv。

流程：
    1. 拉取股票列表，得到 股票代码 -> (简称, orgId) 的映射；查询公告接口必须同时传代码和 orgId。
    2. 按股票查询"年度报告"类公告（可能分多页返回）。
    3. 从公告里挑出每个年份的正式年报全文（排除摘要、英文版、已取消等），见 select_annual_reports。
    4. 下载 PDF 到 data/raw/{doc_id}.pdf，登记到 manifest.csv。

用法：
    uv run python -m ingest.download --codes 600519,000858 --years 2024,2025
    加 --force 会忽略已下载的文件、重新下载

设计要点：
    - 纯函数（标题解析、公告筛选、manifest 读写）与网络 I/O 分离，纯函数可以不连网做单元测试。
    - 下载先写 .part 临时文件，校验 PDF 文件头后再改名，中途失败不会留下"半个文件"。
    - 已存在的有效 PDF 默认跳过，脚本可以反复运行（幂等）。
    - 请求之间有最小间隔、失败有限次重试，避免给巨潮服务器造成压力。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from pathlib import Path
from typing import BinaryIO

import pymupdf

from common.log import setup_logging, timed, trace_context
from config import CninfoSettings, get_settings

logger = logging.getLogger(__name__)

# 巨潮公告分类：沪深两市的"年度报告"
ANNUAL_REPORT_CATEGORY = "category_ndbg_szsh"

# 标题中出现这些词的公告不是我们要的"年报全文"
EXCLUDE_KEYWORDS = ("摘要", "英文", "取消", "港股公告", "更正前", "半年度")
# 同一年份有多份候选时，标题含这些词的优先（通常是更正/修订后的最终版本）
PREFER_KEYWORDS = ("更新后", "更正后", "修订")

# 匹配 "2025年年度报告"、"2025年度报告"、"2025 年年度报告" 等写法，取出年份
_YEAR_RE = re.compile(r"(\d{4})\s*年?\s*年度报告")
# 开启 isHLtitle 后，标题里的搜索关键字会被 <em> 标签包裹，需要去掉
_TAG_RE = re.compile(r"<[^>]+>")

MANIFEST_FIELDS = (
    "doc_id", "company", "stock_code", "year", "file_name", "n_pages",
    "exchange", "publish_date", "source_url", "sha256",
)  # fmt: skip


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StockInfo:
    """一只股票在巨潮的基本信息。"""

    code: str  # 6 位股票代码，保留前导 0，如 "000858"
    name: str  # 简称，如 "五粮液"
    org_id: str  # 巨潮内部机构 ID，如 "gssz0000858"

    @property
    def exchange(self) -> str:
        """根据代码前缀判断交易所。6 开头（含科创板 688）为上交所，0/3 开头为深交所。"""
        if self.code.startswith("6"):
            return "SSE"
        if self.code.startswith(("0", "3")):
            return "SZSE"
        raise ValueError(f"暂不支持的股票代码（北交所等）：{self.code}")

    @property
    def column(self) -> str:
        """公告查询接口的 column 参数：上交所 sse，深交所 szse。"""
        return "sse" if self.exchange == "SSE" else "szse"


@dataclass(frozen=True)
class Announcement:
    """一条公告（已清洗）。"""

    title: str  # 去掉 HTML 标签后的标题
    adjunct_url: str  # PDF 相对路径，如 "finalpage/2025-04-03/1222993920.PDF"
    publish_time: datetime
    size_kb: int | None


@dataclass
class ManifestRow:
    """manifest.csv 的一行。字段顺序与 MANIFEST_FIELDS 一致。"""

    doc_id: str  # {股票代码}_{年份}，全项目统一的文档主键
    company: str
    stock_code: str
    year: int
    file_name: str  # 相对 data/raw 的文件名
    n_pages: int
    exchange: str
    publish_date: str  # YYYY-MM-DD
    source_url: str  # 下载来源，便于追溯
    sha256: str  # 文件指纹，可校验文件是否被改动或损坏


# ---------------------------------------------------------------------------
# 纯函数：标题解析与公告筛选（不涉及网络，可直接单元测试）
# ---------------------------------------------------------------------------


def make_doc_id(stock_code: str, year: int) -> str:
    return f"{stock_code}_{year}"


def clean_title(title: str) -> str:
    """去掉 HTML 标签和首尾空白。"""
    return _TAG_RE.sub("", title).strip()


def extract_report_year(title: str) -> int | None:
    """从年报标题中取出报告所属年份；不是年度报告标题时返回 None。

    >>> extract_report_year("贵州茅台2024年年度报告")
    2024
    >>> extract_report_year("招商银行股份有限公司2025年度报告")
    2025
    """
    m = _YEAR_RE.search(title)
    return int(m.group(1)) if m else None


def parse_announcements(payload: dict) -> list[Announcement]:
    """把公告查询接口返回的 JSON 转成 Announcement 列表。

    接口在没有结果时 announcements 字段为 null，这里统一返回空列表。
    """
    result = []
    for a in payload.get("announcements") or []:
        result.append(
            Announcement(
                title=clean_title(a["announcementTitle"]),
                adjunct_url=a["adjunctUrl"],
                # announcementTime 是毫秒级时间戳
                publish_time=datetime.fromtimestamp(a["announcementTime"] / 1000),
                size_kb=a.get("adjunctSize"),
            )
        )
    return result


def select_annual_reports(anns: list[Announcement], years: set[int]) -> dict[int, Announcement]:
    """从一只股票的公告中，为每个目标年份挑出一份年报全文。

    规则：
        1. 标题能解析出年份、且年份在目标集合内；
        2. 标题不含 EXCLUDE_KEYWORDS（摘要、英文版、已取消、港股公告等）；
        3. 附件是 PDF；
        4. 同一年份剩多份时，先比"标题是否含 PREFER_KEYWORDS"，再比发布时间，取最大者。

    Returns:
        {年份: 公告}。没找到的年份不会出现在结果里，由调用方决定如何报告。
    """
    best: dict[int, Announcement] = {}
    for ann in anns:
        year = extract_report_year(ann.title)
        if year is None or year not in years:
            continue
        if any(k in ann.title for k in EXCLUDE_KEYWORDS):
            continue
        if not ann.adjunct_url.lower().endswith(".pdf"):
            continue
        if year not in best or _rank(ann) > _rank(best[year]):
            best[year] = ann
    return best


def _rank(ann: Announcement) -> tuple[bool, datetime]:
    """排序键：元组按位比较，先比是否为修订版，再比发布时间。"""
    return (any(k in ann.title for k in PREFER_KEYWORDS), ann.publish_time)


def build_query_form(stock: StockInfo, start: date, end: date, page_num: int, page_size: int = 30) -> dict[str, str]:
    """构造公告查询接口的表单参数。"""
    return {
        "stock": f"{stock.code},{stock.org_id}",
        "tabName": "fulltext",
        "pageSize": str(page_size),
        "pageNum": str(page_num),
        "column": stock.column,
        "category": ANNUAL_REPORT_CATEGORY,
        "plate": "",
        "seDate": f"{start.isoformat()}~{end.isoformat()}",
        "searchkey": "",
        "secid": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }


def publish_window(years: set[int]) -> tuple[date, date]:
    """Y 年的年报在 Y+1 年发布，据此确定查询的发布日期范围。"""
    return date(min(years) + 1, 1, 1), date(max(years) + 1, 12, 31)


# ---------------------------------------------------------------------------
# 文件保存与 manifest
# ---------------------------------------------------------------------------


def is_pdf(path: Path) -> bool:
    """通过文件头判断是否为 PDF（PDF 文件以 "%PDF-" 开头）。"""
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        # 分块读取，避免大文件一次性读入内存
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_pdf_stream(src: BinaryIO, dest: Path) -> str:
    """把字节流保存为 PDF，返回 sha256。

    先写 dest.part，确认是 PDF 后再原子地改名为 dest；
    如果内容不是 PDF（例如服务器返回了错误页），删除临时文件并抛出 ValueError。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    try:
        with tmp.open("wb") as f:
            for block in iter(lambda: src.read(1 << 16), b""):
                h.update(block)
                f.write(block)
        if not is_pdf(tmp):
            raise ValueError(f"下载内容不是 PDF：{dest.name}")
        tmp.replace(dest)  # 同一文件系统内的 rename 是原子操作
    finally:
        tmp.unlink(missing_ok=True)  # 成功时 tmp 已被改名，这里什么都不做；失败时清理
    return h.hexdigest()


def count_pages(path: Path) -> int:
    with pymupdf.open(path) as doc:
        return doc.page_count


def read_manifest(path: Path) -> dict[str, ManifestRow]:
    """读取 manifest，返回 {doc_id: 行}；文件不存在时返回空字典。"""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        rows = {}
        for r in csv.DictReader(f):
            # CSV 读出来全是字符串；stock_code 必须保持字符串以保留前导 0
            r["year"] = int(r["year"])
            r["n_pages"] = int(r["n_pages"])
            rows[r["doc_id"]] = ManifestRow(**r)
        return rows


def write_manifest(path: Path, rows: dict[str, ManifestRow]) -> None:
    """按 doc_id 排序写出 manifest（排序保证每次输出稳定，git diff 干净）。"""
    assert tuple(f.name for f in fields(ManifestRow)) == MANIFEST_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for doc_id in sorted(rows):
            w.writerow(asdict(rows[doc_id]))


# ---------------------------------------------------------------------------
# 网络访问
# ---------------------------------------------------------------------------


class CninfoClient:
    """巨潮资讯网的最小客户端：限速 + 重试。"""

    def __init__(self, cfg: CninfoSettings) -> None:
        self.cfg = cfg
        self._last_request = 0.0  # 上一次请求的时间（monotonic 时钟）

    def _throttle(self) -> None:
        """保证相邻两次请求的间隔不小于 request_interval_s。"""
        wait = self.cfg.request_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _open(self, url: str, form: dict[str, str] | None = None):
        """发起请求并返回响应对象；form 不为空时用 POST 表单提交。

        网络错误和 5xx 会按 1s、2s、4s... 退避重试；4xx 属于请求本身有问题，直接抛出。
        """
        data = urllib.parse.urlencode(form).encode() if form is not None else None
        headers = {"User-Agent": "Mozilla/5.0 (report-copilot; personal research)"}
        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
        req = urllib.request.Request(url, data=data, headers=headers)

        for attempt in range(1, self.cfg.max_retries + 1):
            self._throttle()
            try:
                return urllib.request.urlopen(req, timeout=self.cfg.timeout_s)
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == self.cfg.max_retries:
                    raise
                err: Exception = e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.cfg.max_retries:
                    raise
                err = e
            backoff = 2 ** (attempt - 1)
            logger.warning("request retry", extra={"url": url, "attempt": attempt, "backoff_s": backoff, "error": repr(err)})
            time.sleep(backoff)
        raise AssertionError("unreachable")  # 循环内要么 return 要么 raise

    def _json(self, url: str, form: dict[str, str] | None = None) -> dict:
        with self._open(url, form) as resp:
            return json.load(resp)

    def load_stock_index(self) -> dict[str, StockInfo]:
        """拉取股票列表，返回 {股票代码: StockInfo}。"""
        payload = self._json(self.cfg.stock_list_url)
        return {s["code"]: StockInfo(code=s["code"], name=s["zwjc"], org_id=s["orgId"]) for s in payload["stockList"]}

    def query_annual_reports(self, stock: StockInfo, start: date, end: date, max_pages: int = 10) -> list[Announcement]:
        """查询一只股票在日期范围内的全部年度报告类公告（自动翻页）。"""
        anns: list[Announcement] = []
        for page in range(1, max_pages + 1):
            payload = self._json(self.cfg.query_url, build_query_form(stock, start, end, page))
            anns.extend(parse_announcements(payload))
            if not payload.get("hasMore"):
                break
        return anns

    def download(self, adjunct_url: str, dest: Path) -> str:
        """下载 PDF 到 dest，返回 sha256。"""
        url = urllib.parse.urljoin(self.cfg.static_base_url, adjunct_url)
        with self._open(url) as resp:
            return save_pdf_stream(resp, dest)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def fetch_reports(codes: list[str], years: set[int], force: bool = False) -> dict[str, ManifestRow]:
    """下载指定股票、指定年份的年报，更新 manifest，返回本次处理的行。"""
    settings = get_settings()
    raw_dir = settings.paths.raw_dir
    client = CninfoClient(settings.cninfo)
    manifest = read_manifest(settings.paths.manifest_path)

    with timed("load_stock_index", logger):
        index = client.load_stock_index()

    start, end = publish_window(years)
    done: dict[str, ManifestRow] = {}
    for code in codes:
        stock = index.get(code)
        if stock is None:
            logger.error("stock not found", extra={"stock_code": code})
            continue

        with timed("query", logger, stock_code=code) as info:
            chosen = select_annual_reports(client.query_annual_reports(stock, start, end), years)
            info["found_years"] = sorted(chosen)
        for year in sorted(years - chosen.keys()):
            logger.warning("annual report not found", extra={"stock_code": code, "year": year})

        for year, ann in sorted(chosen.items()):
            doc_id = make_doc_id(code, year)
            dest = raw_dir / f"{doc_id}.pdf"
            with timed("download", logger, doc_id=doc_id, title=ann.title, size_kb=ann.size_kb) as info:
                if dest.exists() and is_pdf(dest) and not force:
                    sha = sha256_of(dest)
                    info["skipped"] = True
                else:
                    sha = client.download(ann.adjunct_url, dest)
                    info["skipped"] = False
            row = ManifestRow(
                doc_id=doc_id,
                company=stock.name,
                stock_code=code,
                year=year,
                file_name=dest.name,
                n_pages=count_pages(dest),
                exchange=stock.exchange,
                publish_date=ann.publish_time.date().isoformat(),
                source_url=urllib.parse.urljoin(settings.cninfo.static_base_url, ann.adjunct_url),
                sha256=sha,
            )
            manifest[doc_id] = row
            done[doc_id] = row
            # 每下载一份就落盘一次，中途中断也不会丢掉已完成的登记
            write_manifest(settings.paths.manifest_path, manifest)
    return done


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从巨潮资讯网下载年报 PDF 并更新 manifest.csv")
    p.add_argument("--codes", required=True, help="股票代码，逗号分隔，如 600519,000858")
    p.add_argument("--years", required=True, help="报告年份，逗号分隔，如 2024,2025")
    p.add_argument("--force", action="store_true", help="忽略已下载文件，重新下载")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    years = {int(y) for y in args.years.split(",") if y.strip()}
    with trace_context(), timed("download_all", logger, n_codes=len(codes), years=sorted(years)) as info:
        done = fetch_reports(codes, years, force=args.force)
        info["n_docs"] = len(done)


if __name__ == "__main__":
    main()
