"""留出集出题用的分层抽页：决定"在哪些页出题"，避免出题人挑选对系统有利或不利的位置（docs/decisions.md D-024）。

规则：
    1. 候选页 = 每份年报的全部页，排除：
       - 已有评测集（--against，如 v1、v2）中任一 gold 主位置或 alt 所在的页；
       - 预检字符数少于 min_chars 的页（封面、分隔页等几乎没有正文的页，数据来自 data/parsed/precheck/）；
    2. 每份年报用独立的随机数生成器打乱候选页，取前 n_per_doc 页；
       种子 = f"{seed}:{doc_id}"，所以增删一份年报不会改变其他年报的抽样结果；
    3. 输出按抽样顺序排列（出题时必须按这个顺序使用）。

用法：
    uv run python -m eval.sample_pages --seed 20260928 --n-per-doc 10 \\
        --against eval/goldset.jsonl eval/goldset_v2.jsonl --out eval/v3_sampled_pages.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from config import PROJECT_ROOT, get_settings
from ingest.download import read_manifest


def gold_pages(goldset_paths: list[Path]) -> set[tuple[str, int]]:
    """已有评测集中所有 gold 主位置和 alt 所在的 (doc_id, page)。"""
    pages: set[tuple[str, int]] = set()
    for path in goldset_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            for g in json.loads(line).get("gold", []):
                pages.add((g["doc_id"], g["page"]))
                pages |= {(a["doc_id"], a["page"]) for a in g.get("alt", [])}
    return pages


def page_chars(precheck_dir: Path, doc_id: str) -> dict[int, int]:
    """读取预检逐页明细，返回 {页码: 字符数}。"""
    path = precheck_dir / f"{doc_id}.jsonl"
    return {r["page"]: r["n_chars"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def eligible_pages(
    n_pages: int, chars: dict[int, int], doc_id: str, excluded: set[tuple[str, int]], min_chars: int
) -> list[int]:
    return [p for p in range(1, n_pages + 1) if (doc_id, p) not in excluded and chars.get(p, 0) >= min_chars]


def sample(candidates: list[int], n: int, seed: int, doc_id: str) -> list[int]:
    """用"种子:文档"作为随机数种子打乱候选页，取前 n 页；候选不足 n 页时全部返回。"""
    pages = list(candidates)
    random.Random(f"{seed}:{doc_id}").shuffle(pages)
    return pages[:n]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="留出集分层抽页")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-per-doc", type=int, default=10)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--against", nargs="*", type=Path, default=[], help="已有评测集，其 gold/alt 所在页不参与抽样")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    settings = get_settings()
    manifest = read_manifest(settings.paths.manifest_path)
    excluded = gold_pages([a if a.is_absolute() else PROJECT_ROOT / a for a in args.against])
    precheck_dir = settings.paths.parsed_dir / "precheck"
    result = {}
    for doc_id in sorted(manifest):
        cand = eligible_pages(manifest[doc_id].n_pages, page_chars(precheck_dir, doc_id), doc_id, excluded, args.min_chars)
        result[doc_id] = sample(cand, args.n_per_doc, args.seed, doc_id)
        print(f"{doc_id}: 候选 {len(cand)} 页 -> 抽中 {result[doc_id]}")
    meta = {"seed": args.seed, "n_per_doc": args.n_per_doc, "min_chars": args.min_chars,
            "against": [str(a) for a in args.against], "n_excluded_pages": len(excluded)}  # fmt: skip
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.write_text(json.dumps({"meta": meta, "pages": result}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"排除已有评测集的页 {len(excluded)} 个；结果写入 {args.out}")


if __name__ == "__main__":
    main()
