"""检索评测：对评测集逐题检索，计算指标，写逐题明细，并在 experiments/log.csv 追加一行。

用法：
    uv run python -m eval.run_eval --retriever oracle --no-log            # 合成检索器，检验评测流程
    uv run python -m eval.run_eval --retriever <name> --desc "改动说明"    # 正式实验，写入 log.csv

流程：
    1. 校验评测集 sha256 与配置中冻结的值一致（不一致拒绝运行，保证同版本结果可比）；
    2. 逐题检索（out_of_scope 题没有 gold，跳过并计数），检索深度 = max(k_values)；
    3. 计算命中矩阵与各项指标，按题型分组汇总；
    4. 写逐题明细 experiments/runs/{exp_id}.jsonl（第一行是本次运行的元信息与汇总）；
    5. 追加 log.csv 一行：exp_id 自动递增，git_commit 带 -dirty 标记表示有未提交改动。

接入新检索器：实现 Retriever 协议（name、config、search），在 RETRIEVERS 中注册一个工厂函数。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from common.log import setup_logging, timed, trace_context
from config import PROJECT_ROOT, Settings, get_settings
from eval.metrics import ChunkLike, GoldEvidence, hit_matrix, mean_scores, score_question
from eval.sanity_retrievers import OracleRetriever, RandomPageRetriever
from eval.validate_goldset import PdfPageTexts, load_goldset
from ingest.download import read_manifest

logger = logging.getLogger("eval.run_eval")

# log.csv 中由检索器配置填写的列（来自 retriever.config，缺省为空）
CONFIG_COLUMNS = ("parser", "chunker", "chunk_size", "overlap", "retriever", "fusion", "k_rrf", "reranker", "top_k", "top_n")
# 终端汇总表展示的指标列
DISPLAY_METRICS = ("hit@1", "hit@5", "recall@5", "recall@10", "precision@5", "mrr@10", "ndcg@10")
PREVIEW_CHARS = 80  # 明细中保存的 chunk 文本预览长度（控制文件体积）


class Retriever(Protocol):
    name: str
    config: dict[str, Any]  # parser、chunker、chunk_size 等，写入 log.csv 对应列

    def search(self, query: str, top_k: int) -> Sequence[ChunkLike]: ...


# 检索器注册表：名字 -> 工厂函数(settings, goldset 题目) -> Retriever
RetrieverFactory = Callable[[Settings, list[dict]], Retriever]


def _page_texts(settings: Settings) -> PdfPageTexts:
    return PdfPageTexts(read_manifest(settings.paths.manifest_path), settings.paths.raw_dir)


def _dense(settings: Settings, _items: list[dict]) -> Retriever:
    # 延迟导入：只有真正评测 dense 时才加载 torch / FlagEmbedding / qdrant-client
    from qdrant_client import QdrantClient

    from retrieval.dense import DenseRetriever
    from retrieval.embedding import build_embedder

    return DenseRetriever(
        QdrantClient(url=settings.store.qdrant_url),
        settings.collection_name(),
        build_embedder(settings.embedding, settings.index.max_length),
        settings,
    )


def _pipeline(settings: Settings, _items: list[dict]) -> Retriever:
    from retrieval.search import build_pipeline  # 延迟导入，理由同 _dense

    return build_pipeline(settings)


RETRIEVERS: dict[str, RetrieverFactory] = {
    "oracle": lambda s, items: OracleRetriever(items, _page_texts(s)),
    "random_page": lambda s, items: RandomPageRetriever(read_manifest(s.paths.manifest_path), _page_texts(s)),
    "dense": _dense,  # E001 使用；保留以便复现
    "pipeline": _pipeline,  # 阶段 5 起的实验统一使用，组件由 config.retrieval 的开关决定
}


# ---------------------------------------------------------------------------
# 评测集
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_goldset(path: Path, expected_sha256: str) -> list[dict]:
    """读取评测集，并校验它就是冻结时的那一份文件。"""
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"评测集 sha256 与冻结值不一致：{actual[:12]}… != {expected_sha256[:12]}…。"
            "冻结后的评测集不能修改；如需修改请升级版本并更新 EVAL__GOLDSET_VERSION / EVAL__GOLDSET_SHA256"
        )
    items, issues = load_goldset(path)
    if issues:
        raise ValueError(f"评测集存在无法解析的行：{[str(i) for i in issues]}")
    return items


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------


def evaluate_question(retriever: Retriever, item: dict, ks: Sequence[int]) -> dict:
    """检索一道题并计算指标，返回该题的明细记录。"""
    evidences = [GoldEvidence.from_dict(g) for g in item["gold"]]
    with timed("retrieve", logger, qid=item["id"], retriever=retriever.name) as info:
        chunks = list(retriever.search(item["question"], max(ks)))
        info["n_results"] = len(chunks)
    hits = hit_matrix(chunks, evidences)
    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "n_gold": len(evidences),
        "latency_ms": info["elapsed_ms"],
        "scores": score_question(hits, len(evidences), ks),
        "retrieved": [
            {
                "rank": rank,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "hits": sorted(h),  # 命中的 gold 下标
                "preview": " ".join(c.text.split())[:PREVIEW_CHARS],
            }
            for rank, (c, h) in enumerate(zip(chunks, hits), start=1)
        ],
    }


SUBSETS = ("all", "original", "paraphrase")


def select_subset(items: list[dict], subset: str) -> list[dict]:
    """按 variant 取子集（v1 的题没有 variant 字段，视为 original）。"""
    if subset == "all":
        return items
    return [it for it in items if it.get("variant", "original") == subset]


def evaluate(retriever: Retriever, items: list[dict], ks: Sequence[int]) -> tuple[list[dict], dict]:
    """逐题评测，返回 (逐题记录, 汇总)。out_of_scope 题不参与检索指标。"""
    records = [evaluate_question(retriever, it, ks) for it in items if it["type"] != "out_of_scope"]
    return records, summarize(records, n_skipped=sum(it["type"] == "out_of_scope" for it in items))


def percentile(values: Sequence[float], p: float) -> float | None:
    """最近秩法（nearest-rank）分位数：排序后取第 ceil(p/100 * n) 个值。空列表返回 None。

    样本只有几十个时，插值法与最近秩法差别不大；最近秩法的结果一定是真实观测值，更好解释。
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def summarize(records: list[dict], n_skipped: int = 0) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r["scores"])
    latencies = [r["latency_ms"] for r in records]
    return {
        "n_questions": len(records),
        "n_out_of_scope_skipped": n_skipped,
        "overall": mean_scores([r["scores"] for r in records]),
        "by_type": {t: {"n": len(rows), **mean_scores(rows)} for t, rows in sorted(by_type.items())},
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
    }


# ---------------------------------------------------------------------------
# 实验记录
# ---------------------------------------------------------------------------


def git_commit(cwd: Path = PROJECT_ROOT) -> str:
    """当前提交的短哈希；工作区有未提交改动时加 -dirty；不在 git 仓库中时返回 nogit。

    判断 dirty 时忽略 experiments/：那里是实验的输出（log.csv、逐题明细），不影响检索结果。
    否则连续跑两个实验时，第二个会因为第一个的记录还没提交而被误标为 dirty（E004 就是这种情况）。
    """
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)experiments"],
            cwd=cwd, capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    return head.stdout.strip() + ("-dirty" if status.stdout.strip() else "")


def next_exp_id(log_path: Path) -> str:
    """在 log.csv 已有的 E### 编号基础上加 1；没有记录时从 E001 开始。"""
    nums = [0]
    if log_path.exists():
        with log_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                m = re.fullmatch(r"E(\d+)", row.get("exp_id") or "")
                if m:
                    nums.append(int(m.group(1)))
    return f"E{max(nums) + 1:03d}"


def build_log_row(
    header: Sequence[str], exp_id: str, retriever: Retriever, summary: dict, ks: Sequence[int],
    desc: str, note: str, goldset_version: str, commit: str,
) -> dict[str, str]:  # fmt: skip
    """按 log.csv 表头组装一行；表头里没有的键不会写入。"""
    row = {col: "" for col in header}
    for col in CONFIG_COLUMNS:
        row[col] = str(retriever.config.get(col, ""))
    row["retriever"] = row["retriever"] or retriever.name
    row["top_k"] = row["top_k"] or str(max(ks))
    overall = summary["overall"]
    row.update(
        {
            "exp_id": exp_id,
            "date": date.today().isoformat(),
            "git_commit": commit,
            "改动说明": desc,
            "goldset_version": goldset_version,
            "n_questions": str(summary["n_questions"]),
            "hit@5": f"{overall['hit@5']:.4f}",
            "recall@5": f"{overall['recall@5']:.4f}",
            "mrr@10": f"{overall['mrr@10']:.4f}",
            "备注": "; ".join(x for x in (note, f"明细 experiments/runs/{exp_id}.jsonl") if x),
        }
    )
    return {k: v for k, v in row.items() if k in header}


def append_log(log_path: Path, row: dict[str, str]) -> None:
    """按已有表头追加一行（log.csv 的表头在阶段 0 已建好）。"""
    with log_path.open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    with log_path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writerow(row)


def write_details(path: Path, meta: dict, records: list[dict]) -> None:
    """逐题明细：第一行是元信息与汇总，之后每行一道题。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def format_table(summary: dict) -> str:
    """把总体与分题型指标排成对齐的文本表。"""
    rows = [("overall", summary["n_questions"], summary["overall"])]
    rows += [(t, s["n"], s) for t, s in summary["by_type"].items()]
    head = f"{'':<14}{'n':>4}" + "".join(f"{m:>13}" for m in DISPLAY_METRICS)
    lines = [head]
    for name, n, s in rows:
        lines.append(f"{name:<14}{n:>4}" + "".join(f"{s[m]:>13.4f}" for m in DISPLAY_METRICS))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="检索评测")
    p.add_argument("--retriever", required=True, choices=sorted(RETRIEVERS))
    p.add_argument("--desc", default="", help="改动说明（写入 log.csv；正式实验必填）")
    p.add_argument("--note", default="", help="备注（写入 log.csv）")
    p.add_argument("--no-log", action="store_true", help="不写 log.csv（用于合成检索器或调试）")
    p.add_argument("--goldset", default=None, help="评测集版本，如 v1、v2（默认取 EVAL__GOLDSET_VERSION）")
    p.add_argument("--subset", default="all", choices=SUBSETS, help="按 variant 取子集（v2 起有 paraphrase）")
    args = p.parse_args(argv)
    if not args.no_log and not args.desc:
        p.error("正式实验必须用 --desc 写明改动说明；调试请加 --no-log")

    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    ks = settings.eval.k_values
    version = args.goldset or settings.eval.goldset_version
    if version not in settings.eval.goldsets:
        p.error(f"未登记的评测集版本 {version}，可选：{sorted(settings.eval.goldsets)}")
    frozen = settings.eval.goldsets[version]
    items = select_subset(load_frozen_goldset(settings.eval.goldset_file(version), frozen.sha256), args.subset)
    # 写入 log.csv 的版本标签：整份评测集写 "v2"，子集写 "v2:paraphrase"
    version_label = version if args.subset == "all" else f"{version}:{args.subset}"
    retriever = RETRIEVERS[args.retriever](settings, items)

    log_path = settings.paths.experiments_dir / "log.csv"
    exp_id = f"_nolog_{retriever.name}" if args.no_log else next_exp_id(log_path)
    commit = git_commit()
    with trace_context() as trace_id, timed("eval_all", logger, exp_id=exp_id, retriever=retriever.name):
        records, summary = evaluate(retriever, items, ks)

    meta = {
        "exp_id": exp_id,
        "date": date.today().isoformat(),
        "git_commit": commit,
        "trace_id": trace_id,
        "retriever": retriever.name,
        "retriever_config": retriever.config,
        "goldset_version": version_label,
        "goldset_sha256": frozen.sha256,
        "k_values": ks,
        "summary": summary,
    }
    details_path = settings.paths.experiments_dir / "runs" / f"{exp_id}.jsonl"
    write_details(details_path, meta, records)

    print(format_table(summary))
    lat = summary["latency_ms"]
    shown = details_path.relative_to(PROJECT_ROOT) if details_path.is_relative_to(PROJECT_ROOT) else details_path
    print(f"\n检索题数 {summary['n_questions']}（跳过 out_of_scope {summary['n_out_of_scope_skipped']}）"
          f"  延迟 p50={lat['p50']}ms p95={lat['p95']}ms  明细 {shown}")  # fmt: skip
    if not args.no_log:
        with log_path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        row = build_log_row(header, exp_id, retriever, summary, ks, args.desc, args.note, version_label, commit)
        append_log(log_path, row)
        print(f"已写入 log.csv：{exp_id}  git={commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
