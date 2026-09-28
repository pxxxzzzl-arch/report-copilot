"""多轮对话评测（阶段 7，D-027）：检验 Agent 能否借助对话历史理解省略式追问。

评测集 eval/dialogs_v1.jsonl：每组 2~3 轮，最后一轮是"那五粮液呢？"这类追问；
改写成完整问题后与 v1 的某道已冻结题目（target）等价，参考答案和 gold 直接取自该题。只评最后一轮。

两种模式（对照实验）：
    memory     带记忆：同一对话编号依次问完所有轮次，状态由 PostgresSaver 保存在 Postgres 中
    no_memory  不带记忆：只问最后一轮（新对话，没有历史）

用法（检索配置与生成评测相同）：
    env PARSE__PARSER=v1 EMBEDDING__DEVICE=mps RETRIEVAL__USE_FILTER=true RETRIEVAL__STRIP_ENTITIES=true \\
        RETRIEVAL__SOURCES='["dense","bm25","sparse"]' \\
        uv run python -m eval.run_dialog_eval --mode memory --desc "D001 ..."
    uv run python -m eval.compare_runs D002 D001        # 配对比较（与生成实验相同的指标）

结果写入 experiments/gen_runs/D###.jsonl（格式与 G### 相同，可以直接用 compare_runs 比较），汇总追加到 experiments/dialog_log.csv。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import uuid
from datetime import date

from common.log import setup_logging, timed, trace_context
from config import PROJECT_ROOT, Settings, get_settings
from eval.run_eval import git_commit, load_frozen_goldset, next_exp_id
from eval.run_gen_eval import agent_output, fmt, print_summary, score_and_judge, summarize
from eval.validate_goldset import PdfPageTexts
from generation.llm import Price
from ingest.download import read_manifest

logger = logging.getLogger("eval.run_dialog_eval")

DIALOG_LOG_HEADER = (
    "exp_id,date,git_commit,改动说明,mode,checkpointer,prompt_version,dialog_set,n_dialogs,"
    "judge_correct,num_match,cite_recall,refused_ok,cost_usd,备注"
).split(",")


def load_dialogs(path, sha256: str) -> list[dict]:
    """读取对话集并校验指纹（与 load_frozen_goldset 相同的冻结机制）。"""
    return load_frozen_goldset(path, sha256)


def build_items(dialogs: list[dict], targets: dict[str, dict]) -> list[dict]:
    """把每组对话转成一道"题"：题号用对话编号，问题是最后一轮原话，类型、参考答案、gold 来自 target 题目。"""
    items = []
    for d in dialogs:
        t = targets[d["target"]]
        items.append({**t, "id": d["id"], "question": d["turns"][-1], "target": d["target"], "turns": d["turns"]})
    return items


def run(settings: Settings, dialogs: list[dict], targets: dict[str, dict], mode: str, retriever, checkpointer) -> list[dict]:
    from agent.fin_db import PgFinStore
    from agent.graph import build_deps, build_graph, run_turn

    gen_price = Price(settings.llm.price_input_cache_hit, settings.llm.price_input_cache_miss, settings.llm.price_output)
    store = PgFinStore(settings.store.postgres_dsn.get_secret_value())
    audit = store.log_tool_call if settings.agent.log_tool_calls else None
    deps = build_deps(settings, retriever.search, store, audit=audit)  # 检索与页面文本在 build_deps 内加锁
    graph = build_graph(deps, checkpointer=checkpointer)
    items = build_items(dialogs, targets)

    outputs, turns_log = {}, {}
    try:
        # 顺序执行：对话数少，而且每组对话内部的轮次本来就必须按顺序进行
        for d, it in zip(dialogs, items):
            thread = f"eval-{d['id']}-{uuid.uuid4().hex[:8]}"  # 每次运行用新的对话编号，避免读到上次运行留下的历史
            turns = d["turns"] if mode == "memory" else d["turns"][-1:]
            log = []
            for q in turns:
                with trace_context(), timed("dialog_turn", logger, dialog=d["id"]) as t:
                    st = run_turn(graph, q, settings.agent.max_steps, thread)
                log.append({"question": q, "standalone": st["standalone"], "answer": st["answer"]["answer"]})
            outputs[it["id"]] = agent_output(st, t["elapsed_ms"], settings, gen_price)  # 只评最后一轮
            turns_log[it["id"]] = log
    finally:
        store.close()

    page_texts = PdfPageTexts(read_manifest(settings.paths.manifest_path), settings.paths.raw_dir)
    records = score_and_judge(settings, items, outputs, page_texts)
    for rec, it in zip(records, items):
        rec["target"], rec["turns"] = it["target"], turns_log[it["id"]]
    return records


def append_log(path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DIALOG_LOG_HEADER)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in DIALOG_LOG_HEADER})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="多轮对话评测")
    p.add_argument("--mode", required=True, choices=("memory", "no_memory"))
    p.add_argument("--dialogs", default="dialogs_v1")
    p.add_argument("--checkpointer", default="postgres", choices=("postgres", "memory"),
                   help="postgres：PostgresSaver（默认，验证真实的持久化）；memory：进程内存")
    p.add_argument("--desc", default="")
    p.add_argument("--note", default="")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args(argv)
    if not args.no_log and not args.desc:
        p.error("正式实验必须用 --desc 写明改动说明")

    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    frozen = settings.eval.dialogs[args.dialogs]
    dialogs = load_dialogs(PROJECT_ROOT / frozen.path, frozen.sha256)
    v1 = settings.eval.goldsets["v1"]
    targets = {it["id"]: it for it in load_frozen_goldset(settings.eval.goldset_file("v1"), v1.sha256)}

    from retrieval.search import build_pipeline  # 延迟导入：会加载嵌入模型

    retriever = build_pipeline(settings)
    log_path = settings.paths.experiments_dir / "dialog_log.csv"
    exp_id = "_nolog_dialog" if args.no_log else next_exp_id(log_path, prefix="D")
    commit = git_commit()

    with trace_context(), timed("dialog_eval_all", logger, exp_id=exp_id, n=len(dialogs)):
        if args.checkpointer == "postgres":
            from langgraph.checkpoint.postgres import PostgresSaver

            with PostgresSaver.from_conn_string(settings.store.postgres_dsn.get_secret_value()) as saver:
                saver.setup()
                records = run(settings, dialogs, targets, args.mode, retriever, saver)
        else:
            from langgraph.checkpoint.memory import InMemorySaver

            records = run(settings, dialogs, targets, args.mode, retriever, InMemorySaver())
    summary = summarize(records)

    meta = {
        "exp_id": exp_id, "date": date.today().isoformat(), "git_commit": commit, "mode": args.mode,
        "checkpointer": args.checkpointer, "retrieval_config": retriever.config,
        "agent": settings.agent.model_dump(mode="json"), "llm_model": settings.llm.model,
        "judge_model": settings.judge.model, "judge_prompt_version": settings.judge.prompt_version,
        # compare_runs 用这两个字段确认两次实验用的是同一个评测集
        "goldset_version": args.dialogs, "goldset_sha256": frozen.sha256, "summary": summary,
    }  # fmt: skip
    out = settings.paths.experiments_dir / "gen_runs" / f"{exp_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_summary(summary)
    for r in records:
        mark = {2: "✓", 1: "△", 0: "✗"}.get(r["judge"]["correctness"], "?")
        print(f"{mark} {r['id']}（{r['target']}）改写：{r['turns'][-1]['standalone']}")
    print(f"明细 {out.relative_to(PROJECT_ROOT)}")
    if not args.no_log:
        o = summary["overall"]
        refused_ok = sum(r["metrics"]["refused_ok"] for r in records) / len(records)
        append_log(log_path, {
            "exp_id": exp_id, "date": meta["date"], "git_commit": commit, "改动说明": args.desc, "mode": args.mode,
            "checkpointer": args.checkpointer, "prompt_version": settings.agent.prompt_version,
            "dialog_set": args.dialogs, "n_dialogs": o["n"], "judge_correct": fmt(o["judge_correct"]),
            "num_match": fmt(o["num_match"]), "cite_recall": fmt(o["cite_recall"]), "refused_ok": fmt(refused_ok),
            "cost_usd": summary["cost_usd"],
            "备注": "; ".join(x for x in (args.note, f"明细 experiments/gen_runs/{exp_id}.jsonl") if x),
        })  # fmt: skip
        print(f"已写入 dialog_log.csv：{exp_id}  git={commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
