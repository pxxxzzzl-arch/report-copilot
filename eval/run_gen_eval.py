"""生成评测：检索 -> 作答（带引用）-> 引用校验 -> 规则判分 + LLM 评审 -> 汇总与记录。

用法（检索配置沿用环境变量，例如阶段 5 的最终配置 E013）：
    env PARSE__PARSER=v1 EMBEDDING__DEVICE=mps RETRIEVAL__USE_FILTER=true RETRIEVAL__STRIP_ENTITIES=true \\
        RETRIEVAL__SOURCES='["dense","bm25","sparse"]' \\
        uv run python -m eval.run_gen_eval --goldset v2 --desc "G001 基线"
    加 --limit 5 --no-log 可以先用 5 道题做冒烟测试

流程：
    1. 检索（顺序执行：嵌入模型不是线程安全的）：每题取前 context_k 个 chunk；
    2. 作答（并发 concurrency 路）：deepseek 按提示词输出 JSON（答案 + 引用 + 是否拒答）；
    3. 引用校验：摘录是否真的出现在对应证据中，并定位页码（generation/citation.py）；
    4. 规则判分：数值是否正确、引用是否覆盖 gold 所在页、拒答是否正确（eval/gen_metrics.py）；
    5. LLM 评审（并发）：correctness 0/1/2 与 faithful（eval/judge.py）；
    6. 逐题明细写入 experiments/gen_runs/{G###}.jsonl，汇总追加到 experiments/gen_log.csv。

指标口径：
    judge_correct   评审分数 correctness/2 的平均值（全部题目，含拒答题）—— 主指标
    num_match       关键数值全部命中的题目占比（表格数值、计算、比较、多跳题中参考答案含数值的题）
    faithfulness    非拒答回答中，评审判为 faithful 的比例
    cite_verified   非拒答回答的全部引用中，摘录通过核验的比例（衡量是否编造引用）
    cite_recall     可回答题中，gold 证据所在页被"通过核验的引用"覆盖的比例（拒答记 0）
    refusal_recall  应拒答题（out_of_scope）中实际拒答的比例
    false_refusal   可回答题中错误拒答的比例
    judge_rule_agree 数值题上"评审判满分"与"规则判全部数值命中"一致的比例（评审本身的可信度）
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date
from pathlib import Path

from common.log import setup_logging, timed, trace_context
from config import PROJECT_ROOT, Settings, get_settings
from eval.gen_metrics import citation_recall, numbers_match
from eval.judge import judge_answer, rule_verdict
from eval.run_eval import git_commit, load_frozen_goldset, next_exp_id, percentile, select_subset
from eval.validate_goldset import PdfPageTexts
from generation.answer import generate_answer
from generation.citation import check_citations
from generation.llm import LLMClient, Price
from ingest.download import read_manifest

logger = logging.getLogger("eval.run_gen_eval")

NUMERIC_TYPES = ("table_number", "calc", "compare", "multi_hop")
GEN_LOG_HEADER = (
    "exp_id,date,git_commit,改动说明,retrieval,context_k,llm_model,thinking,prompt_version,judge_model,"
    "goldset_version,n_questions,judge_correct,num_match,faithfulness,cite_verified,cite_recall,"
    "refusal_recall,false_refusal,cost_usd,备注"
).split(",")


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def score_record(item: dict, rec: dict) -> dict:
    """对一题计算规则类指标，写回 rec["metrics"]。rec 需已有 answer / refused / citations。"""
    oos = item["type"] == "out_of_scope"
    refused = rec["refused"]
    cited = set()
    for c in rec["citations"]:
        if c["verified"]:
            if c["page"] is not None:
                cited.add((c["doc_id"], c["page"]))
            else:
                cited |= {(c["doc_id"], p) for p in range(c["page_start"], c["page_end"] + 1)}
    m: dict = {"oos": oos}
    m["refused_ok"] = refused if oos else not refused
    if not oos:
        m["cite_recall"] = 0.0 if refused else citation_recall(item["gold"], cited)
    if not refused:
        m["n_cites"] = len(rec["citations"])
        m["n_verified"] = sum(c["verified"] for c in rec["citations"])
    if item["type"] in NUMERIC_TYPES:
        hit, total = numbers_match(item["answer"], "" if refused else rec["answer"])
        if total:
            m["num_hit"], m["num_total"], m["num_match"] = hit, total, hit == total
    rec["metrics"] = m
    return rec


def summarize(records: list[dict]) -> dict:
    def block(rs: list[dict]) -> dict:
        j = [r["judge"]["correctness"] / 2 for r in rs if r["judge"]["ok"]]
        answered = [r for r in rs if not r["refused"] and r["format_ok"]]
        faithful = [float(r["judge"]["faithful"]) for r in answered if r["judge"]["ok"]]
        n_cites = sum(r["metrics"].get("n_cites", 0) for r in answered)
        n_ver = sum(r["metrics"].get("n_verified", 0) for r in answered)
        num = [float(r["metrics"]["num_match"]) for r in rs if "num_match" in r["metrics"]]
        oos = [r for r in rs if r["metrics"]["oos"]]
        ans_q = [r for r in rs if not r["metrics"]["oos"]]
        # 只统计由大模型评审的题：规则判定的题（如拒答）两者天然一致，计入会虚高一致率
        agree = [
            float((r["judge"]["correctness"] == 2) == r["metrics"]["num_match"])
            for r in rs
            if "num_match" in r["metrics"] and r["judge"]["ok"] and not r["judge"].get("by_rule")
        ]
        return {
            "n": len(rs),
            "judge_correct": _mean(j),
            "num_match": _mean(num),
            "faithfulness": _mean(faithful),
            "cite_verified": n_ver / n_cites if n_cites else None,
            "cite_recall": _mean([r["metrics"]["cite_recall"] for r in ans_q]),
            "refusal_recall": _mean([float(r["refused"]) for r in oos]),
            "false_refusal": _mean([float(r["refused"]) for r in ans_q]),
            "format_error": _mean([float(not r["format_ok"]) for r in rs]),
            "judge_fail": sum(not r["judge"]["ok"] for r in rs),
            "judge_rule_agree": _mean(agree),
            "n_judge_rule": len(agree),
        }

    by_type, by_variant = defaultdict(list), defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
        by_variant[r["variant"]].append(r)
    return {
        "overall": block(records),
        "by_variant": {k: block(v) for k, v in sorted(by_variant.items())},
        "by_type": {k: block(v) for k, v in sorted(by_type.items())},
        "gen_latency_ms": {
            "p50": percentile([r["gen"]["latency_ms"] for r in records if not r["gen"]["from_cache"]], 50),
            "p95": percentile([r["gen"]["latency_ms"] for r in records if not r["gen"]["from_cache"]], 95),
        },
        "cost_usd": round(sum(r["gen"]["cost_usd"] + r["judge"]["cost_usd"] for r in records), 4),
    }


def run(settings: Settings, items: list[dict], retriever) -> list[dict]:
    gcfg = settings.generation
    gen_llm = LLMClient(settings.llm, gcfg.cache_dir)
    judge_llm = LLMClient(settings.llm, gcfg.cache_dir, model=settings.judge.model)
    gen_price = Price(settings.llm.price_input_cache_hit, settings.llm.price_input_cache_miss, settings.llm.price_output)
    judge_price = Price(settings.judge.price_input_cache_hit, settings.judge.price_input_cache_miss, settings.judge.price_output)
    page_texts = PdfPageTexts(read_manifest(settings.paths.manifest_path), settings.paths.raw_dir)

    # 1. 检索（顺序）
    contexts = {}
    for it in items:
        with timed("retrieve", logger, qid=it["id"]):
            contexts[it["id"]] = retriever.search(it["question"], gcfg.context_k)

    # 2. 作答（并发）；每道题在自己的 trace 里执行，日志可按 trace_id 串起来
    def gen_one(it: dict):
        with trace_context(), timed("generate", logger, qid=it["id"]):
            return generate_answer(it["question"], contexts[it["id"]], gen_llm, gcfg.max_tokens)

    with ThreadPoolExecutor(gcfg.concurrency) as pool:
        answers = dict(zip([it["id"] for it in items], pool.map(gen_one, items)))

    # 3~4. 引用校验与规则判分（主线程，页面文本缓存非线程安全）
    records = []
    for it in items:
        ans, res = answers[it["id"]]
        chunks = contexts[it["id"]]
        checks = check_citations(ans.citations, chunks, page_texts)
        rec = {
            "id": it["id"], "type": it["type"], "variant": it.get("variant", "original"),
            "question": it["question"], "reference": it["answer"],
            "context": [{"rank": i, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "page_start": c.page_start,
                         "page_end": c.page_end} for i, c in enumerate(chunks, start=1)],
            "answer": ans.answer, "refused": ans.refused, "format_ok": ans.format_ok, "error": ans.error,
            "n_bad_citations": ans.n_bad_citations, "citations": [asdict(c) for c in checks],
            "gen": {"latency_ms": res.latency_ms, "from_cache": res.from_cache, "usage": asdict(res.usage),
                    "cost_usd": 0.0 if res.from_cache else round(gen_price.cost(res.usage), 6)},
        }  # fmt: skip
        records.append(score_record(it, rec))

    # 5. 评审（并发）：只把系统实际引用的证据（有效编号）交给评审
    def judge_one(rec: dict):
        answered = rec["format_ok"] and bool(rec["answer"].strip())
        ruled = rule_verdict(rec["metrics"]["oos"], rec["refused"], answered)
        if ruled is not None:  # 拒答、未作答：按规则判定，不调用大模型
            return ruled, None
        chunks = contexts[rec["id"]]
        refs = sorted({c["ref"] for c in rec["citations"] if c["valid_ref"]})
        with trace_context(), timed("judge", logger, qid=rec["id"]):
            return judge_answer(judge_llm, settings.judge.max_tokens, rec["question"], rec["reference"],
                                rec["answer"], rec["refused"], [chunks[r - 1].text for r in refs])  # fmt: skip

    with ThreadPoolExecutor(gcfg.concurrency) as pool:
        verdicts = list(pool.map(judge_one, records))
    for rec, (v, res) in zip(records, verdicts):
        cost = 0.0 if res is None or res.from_cache else round(judge_price.cost(res.usage), 6)
        rec["judge"] = {**asdict(v), "cost_usd": cost}
    return records


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def print_summary(summary: dict) -> None:
    cols = ("judge_correct", "num_match", "faithfulness", "cite_verified", "cite_recall", "refusal_recall", "false_refusal")
    print(f"{'':<14}{'n':>4}" + "".join(f"{c:>15}" for c in cols))
    rows = [("overall", summary["overall"])] + [(f"[{k}]", v) for k, v in summary["by_variant"].items()]
    rows += list(summary["by_type"].items())
    for name, b in rows:
        print(f"{name:<14}{b['n']:>4}" + "".join(f"{fmt(b[c]) or '-':>15}" for c in cols))
    o = summary["overall"]
    print(f"\n格式错误率 {fmt(o['format_error'])}  评审失败 {o['judge_fail']} 题  "
          f"评审与规则一致率 {fmt(o['judge_rule_agree'])}（{o['n_judge_rule']} 题）  "
          f"生成延迟 p50={summary['gen_latency_ms']['p50']}ms  本次费用 ${summary['cost_usd']}")  # fmt: skip


def append_gen_log(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GEN_LOG_HEADER)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in GEN_LOG_HEADER})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="生成评测")
    p.add_argument("--goldset", default=None)
    p.add_argument("--subset", default="all", choices=("all", "original", "paraphrase"))
    p.add_argument("--desc", default="")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="只评测前 N 道题（冒烟测试用，必须配合 --no-log）")
    args = p.parse_args(argv)
    if args.limit and not args.no_log:
        p.error("--limit 只用于冒烟测试，必须同时加 --no-log")
    if not args.no_log and not args.desc:
        p.error("正式实验必须用 --desc 写明改动说明")

    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    version = args.goldset or settings.eval.goldset_version
    frozen = settings.eval.goldsets[version]
    items = select_subset(load_frozen_goldset(settings.eval.goldset_file(version), frozen.sha256), args.subset)
    if args.limit:
        items = items[: args.limit]

    from retrieval.search import build_pipeline  # 延迟导入：会加载嵌入模型

    retriever = build_pipeline(settings)
    log_path = settings.paths.experiments_dir / "gen_log.csv"
    exp_id = "_nolog_gen" if args.no_log else next_exp_id(log_path, prefix="G")
    commit = git_commit()
    with trace_context(), timed("gen_eval_all", logger, exp_id=exp_id, n=len(items)):
        records = run(settings, items, retriever)
    summary = summarize(records)

    version_label = version if args.subset == "all" else f"{version}:{args.subset}"
    meta = {
        "exp_id": exp_id, "date": date.today().isoformat(), "git_commit": commit,
        "retrieval_config": retriever.config, "generation": settings.generation.model_dump(mode="json"),
        "llm_model": settings.llm.model, "thinking": settings.llm.thinking, "temperature": settings.llm.temperature,
        "judge_model": settings.judge.model, "judge_prompt_version": settings.judge.prompt_version,
        "goldset_version": version_label, "goldset_sha256": frozen.sha256, "summary": summary,
    }  # fmt: skip
    out = settings.paths.experiments_dir / "gen_runs" / f"{exp_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_summary(summary)
    shown = out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out
    print(f"明细 {shown}")
    if not args.no_log:
        o = summary["overall"]
        append_gen_log(log_path, {
            "exp_id": exp_id, "date": meta["date"], "git_commit": commit, "改动说明": args.desc,
            "retrieval": retriever.config.get("retriever", ""), "context_k": settings.generation.context_k,
            "llm_model": settings.llm.model, "thinking": settings.llm.thinking,
            "prompt_version": settings.generation.prompt_version, "judge_model": settings.judge.model,
            "goldset_version": version_label, "n_questions": o["n"], "judge_correct": fmt(o["judge_correct"]),
            "num_match": fmt(o["num_match"]), "faithfulness": fmt(o["faithfulness"]),
            "cite_verified": fmt(o["cite_verified"]), "cite_recall": fmt(o["cite_recall"]),
            "refusal_recall": fmt(o["refusal_recall"]), "false_refusal": fmt(o["false_refusal"]),
            "cost_usd": summary["cost_usd"], "备注": "; ".join(x for x in (args.note, f"明细 experiments/gen_runs/{exp_id}.jsonl") if x),
        })  # fmt: skip
        print(f"已写入 gen_log.csv：{exp_id}  git={commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
