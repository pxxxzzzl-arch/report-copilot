"""评测集校验：逐题回到 PDF 原文，检查标注是否真实、格式是否合规。

检查项（error 会导致退出码 1，warning 只提示）：
    格式    每行合法 JSON；必填字段齐全；id 唯一且形如 q001；type 合法
    gold    out_of_scope 题 gold 必须为空，其他题必须非空；
            multi_hop / calc / compare 至少 2 条 gold；compare 题的 gold 应来自至少 2 份文档
    位置    doc_id 在 manifest 中；page 在 [1, n_pages] 内（alt 中的位置同样检查）
    原文    归一化后的 span 必须出现在归一化后的该页文本中（主位置和每个 alt 位置都要满足）
    改写    variant=paraphrase 的题（v2 起）：source_id 必须指向一道原题，题型、答案、gold 与原题完全一致，
            问题文字必须不同；问题里出现 gold 的 span 原文时给出 warning（改写不应"泄露"答案片段）
    提示    span 过短（易误命中）；span 在文档其他页也出现、但未列入 alt（用 --suggest-alt 查看）

用法：
    uv run python -m eval.validate_goldset                  # 校验 eval/goldset.jsonl
    uv run python -m eval.validate_goldset --suggest-alt    # 额外列出每条 gold 的 span 在同一公司各年报中出现的所有页
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf

from common.text import normalize
from config import PROJECT_ROOT, get_settings
from ingest.download import ManifestRow, read_manifest

QUESTION_TYPES = ("fact", "table_number", "multi_hop", "calc", "compare", "out_of_scope")
VARIANTS = ("original", "paraphrase")  # v2 起的可选字段 variant；缺省视为 original
REQUIRED_FIELDS = ("id", "type", "question", "answer", "gold")
MULTI_EVIDENCE_TYPES = ("multi_hop", "calc", "compare")  # 需要至少 2 条证据的题型
MIN_SPAN_CHARS = 4  # 归一化后短于此长度的 span 很容易在无关文本中误命中
MIN_QUESTIONS_PER_DOC = 3
_ID_RE = re.compile(r"^q\d{3}$")

DEFAULT_GOLDSET = PROJECT_ROOT / "eval" / "goldset.jsonl"

# 按 doc_id 返回该文档每一页归一化后的文本（下标 0 对应第 1 页）
PageTextLoader = Callable[[str], list[str]]


@dataclass
class Issue:
    qid: str  # 题目 id；整行解析失败时为 "line N"
    level: Literal["error", "warning"]
    msg: str

    def __str__(self) -> str:
        return f"[{self.level.upper():7}] {self.qid}: {self.msg}"


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    type_counts: Counter = field(default_factory=Counter)
    doc_counts: Counter = field(default_factory=Counter)  # 每份文档被多少道题引用
    n_questions: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]


# ---------------------------------------------------------------------------
# 页面文本
# ---------------------------------------------------------------------------


class PdfPageTexts:
    """按需读取并缓存每份 PDF 的逐页归一化文本（同一文档只读一次）。"""

    def __init__(self, manifest: dict[str, ManifestRow], raw_dir: Path) -> None:
        self.manifest = manifest
        self.raw_dir = raw_dir
        self._cache: dict[str, list[str]] = {}

    def __call__(self, doc_id: str) -> list[str]:
        if doc_id not in self._cache:
            path = self.raw_dir / self.manifest[doc_id].file_name
            with pymupdf.open(path) as doc:
                self._cache[doc_id] = [normalize(p.get_text()) for p in doc]
        return self._cache[doc_id]


def find_span_pages(span: str, pages: list[str]) -> list[int]:
    """返回归一化 span 出现的所有页码（从 1 开始）。"""
    s = normalize(span)
    return [i for i, text in enumerate(pages, start=1) if s and s in text]


# ---------------------------------------------------------------------------
# 单题校验
# ---------------------------------------------------------------------------


def _check_location(qid: str, loc: dict, manifest: dict[str, ManifestRow], what: str) -> list[Issue]:
    """检查 doc_id 与 page 是否有效。what 用于错误信息里说明是主位置还是 alt。"""
    doc_id, page = loc.get("doc_id"), loc.get("page")
    if doc_id not in manifest:
        return [Issue(qid, "error", f"{what} doc_id 不在 manifest 中：{doc_id!r}")]
    if not isinstance(page, int) or not 1 <= page <= manifest[doc_id].n_pages:
        return [Issue(qid, "error", f"{what} page 越界：{doc_id} p{page}（共 {manifest[doc_id].n_pages} 页）")]
    return []


def check_item(item: dict, manifest: dict[str, ManifestRow], page_texts: PageTextLoader) -> list[Issue]:
    """校验一道题，返回发现的问题。"""
    qid = str(item.get("id", "?"))
    missing = [f for f in REQUIRED_FIELDS if f not in item]
    if missing:
        return [Issue(qid, "error", f"缺少字段 {missing}")]

    issues: list[Issue] = []
    if not _ID_RE.match(qid):
        issues.append(Issue(qid, "error", "id 应形如 q001"))
    qtype = item["type"]
    if qtype not in QUESTION_TYPES:
        return issues + [Issue(qid, "error", f"未知题型 {qtype!r}")]
    if not str(item["question"]).strip() or not str(item["answer"]).strip():
        issues.append(Issue(qid, "error", "question / answer 不能为空"))

    gold = item["gold"]
    if not isinstance(gold, list):
        return issues + [Issue(qid, "error", "gold 必须是列表")]
    if qtype == "out_of_scope":
        if gold:
            issues.append(Issue(qid, "error", "out_of_scope 题的 gold 必须为空"))
        return issues
    if not gold:
        return issues + [Issue(qid, "error", "非 out_of_scope 题的 gold 不能为空")]
    if qtype in MULTI_EVIDENCE_TYPES and len(gold) < 2:
        issues.append(Issue(qid, "error", f"{qtype} 题至少需要 2 条 gold"))
    if qtype == "compare" and len({g.get("doc_id") for g in gold}) < 2:
        issues.append(Issue(qid, "warning", "compare 题的 gold 只来自 1 份文档"))

    for gi, g in enumerate(gold):
        tag = f"gold[{gi}]"
        span = g.get("span")
        if not isinstance(span, str) or not normalize(span):
            issues.append(Issue(qid, "error", f"{tag} span 为空"))
            continue
        if len(normalize(span)) < MIN_SPAN_CHARS:
            issues.append(Issue(qid, "warning", f"{tag} span 过短，易误命中：{span!r}"))

        # 主位置 + 所有 alt 位置，都必须有效且包含 span
        locations = [(tag, g)] + [(f"{tag}.alt[{ai}]", a) for ai, a in enumerate(g.get("alt", []))]
        seen: set[tuple[str, int]] = set()
        for what, loc in locations:
            loc_issues = _check_location(qid, loc, manifest, what)
            if loc_issues:
                issues.extend(loc_issues)
                continue
            key = (loc["doc_id"], loc["page"])
            if key in seen:
                issues.append(Issue(qid, "error", f"{what} 与前面的位置重复：{key}"))
            seen.add(key)
            if normalize(span) not in page_texts(loc["doc_id"])[loc["page"] - 1]:
                issues.append(Issue(qid, "error", f"{what} 在 {loc['doc_id']} p{loc['page']} 找不到 span {span!r}"))

        # 同一文档内 span 还出现在其他页、却没列入 alt：提示人工确认是否为等价位置
        if g.get("doc_id") not in manifest:  # 主位置无效时上面已报 error，这里不再读取文档
            continue
        others = [p for p in find_span_pages(span, page_texts(g["doc_id"])) if (g["doc_id"], p) not in seen]
        if others:
            issues.append(Issue(qid, "warning", f"{tag} span 还出现在 {g['doc_id']} 的第 {others} 页，确认是否需要加入 alt"))
    return issues


def check_variant(item: dict, by_id: dict[str, dict]) -> list[Issue]:
    """校验改写题与原题的对应关系。"""
    qid = str(item.get("id", "?"))
    variant = item.get("variant", "original")
    if variant not in VARIANTS:
        return [Issue(qid, "error", f"未知 variant {variant!r}")]
    if variant == "original":
        return []
    src = by_id.get(str(item.get("source_id")))
    if src is None or src.get("variant", "original") != "original":
        return [Issue(qid, "error", f"source_id {item.get('source_id')!r} 不是评测集中的原题")]
    issues = [
        Issue(qid, "error", f"改写题的 {f} 必须与原题 {src['id']} 一致")
        for f in ("type", "answer", "gold")
        if item.get(f) != src.get(f)
    ]
    if normalize(item["question"]) == normalize(src["question"]):
        issues.append(Issue(qid, "error", "改写题的问题文字与原题相同"))
    leaked = [g["span"] for g in item.get("gold", []) if normalize(g["span"]) in normalize(item["question"])]
    if leaked:
        issues.append(Issue(qid, "warning", f"改写后的问题包含 gold 原文片段 {leaked}"))
    return issues


# ---------------------------------------------------------------------------
# 整体校验
# ---------------------------------------------------------------------------


def load_goldset(path: Path) -> tuple[list[dict], list[Issue]]:
    """逐行读取 JSONL；空行跳过，解析失败的行记为 error。"""
    items, issues = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            issues.append(Issue(f"line {n}", "error", f"JSON 解析失败：{e}"))
    return items, issues


def validate(items: list[dict], manifest: dict[str, ManifestRow], page_texts: PageTextLoader) -> Report:
    report = Report(n_questions=len(items))
    ids = Counter(str(it.get("id")) for it in items)
    for qid, n in ids.items():
        if n > 1:
            report.issues.append(Issue(qid, "error", f"id 重复 {n} 次"))

    by_id = {str(it.get("id")): it for it in items}
    for it in items:
        report.issues.extend(check_item(it, manifest, page_texts))
        report.issues.extend(check_variant(it, by_id))
        report.type_counts[it.get("type")] += 1
        for doc_id in {g.get("doc_id") for g in it.get("gold") or []}:
            report.doc_counts[doc_id] += 1

    for doc_id in sorted(manifest):
        if report.doc_counts[doc_id] < MIN_QUESTIONS_PER_DOC:
            report.issues.append(
                Issue("-", "warning", f"{doc_id} 只被 {report.doc_counts[doc_id]} 道题引用（建议 ≥{MIN_QUESTIONS_PER_DOC}）")
            )
    return report


def suggest_alt(items: list[dict], manifest: dict[str, ManifestRow], page_texts: PageTextLoader) -> None:
    """打印每条 gold 的 span 在同一公司所有年报中出现的页，辅助人工决定 alt。"""
    for it in items:
        for gi, g in enumerate(it.get("gold") or []):
            code = str(g.get("doc_id", "")).split("_")[0]
            hits = {
                d: find_span_pages(g["span"], page_texts(d))
                for d in sorted(manifest)
                if d.startswith(code + "_")
            }
            hits = {d: p for d, p in hits.items() if p}
            print(f"{it['id']} gold[{gi}] {g['span']!r} 标注={g['doc_id']} p{g['page']}  出现于={hits}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="校验评测集")
    p.add_argument("--path", type=Path, default=DEFAULT_GOLDSET)
    p.add_argument("--suggest-alt", action="store_true", help="列出每条 span 在同公司各年报中出现的所有页")
    p.add_argument("--quiet", action="store_true", help="不打印 warning")
    args = p.parse_args(argv)

    settings = get_settings()
    manifest = read_manifest(settings.paths.manifest_path)
    page_texts = PdfPageTexts(manifest, settings.paths.raw_dir)

    items, load_issues = load_goldset(args.path)
    report = validate(items, manifest, page_texts)
    report.issues[:0] = load_issues

    if args.suggest_alt:
        suggest_alt(items, manifest, page_texts)
    for issue in report.issues:
        if issue.level == "error" or not args.quiet:
            print(issue)
    print(f"\n题目数：{report.n_questions}  题型分布：{dict(report.type_counts)}")
    print(f"文档引用次数：{dict(sorted(report.doc_counts.items()))}")
    print(f"error：{len(report.errors)}  warning：{len(report.warnings)}")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
