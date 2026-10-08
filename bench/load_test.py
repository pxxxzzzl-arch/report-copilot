"""阶段 11 轻量压测：对运行中的 API 服务（阶段 10 的容器）施加并发，记录延迟、吞吐、错误和服务端耗时构成（D-032）。

两种模式：
    闭环并发（默认）：N 个工作线程，每个线程问完一题马上问下一题，直到本组题目问完。
        uv run python -m bench.load_test --concurrency 4 --group 2 --desc "并发 4"
    突发过载：每轮同时发出 --burst 个请求（用 Barrier 对齐），共 --rounds 轮；
        之后再同时发 --verify 个请求，检查并发名额没有泄漏。
        uv run python -m bench.load_test --burst 12 --rounds 3 --verify 8 --group 0 --desc "过载突发"

请求走 /v1/chat/stream（前端实际使用的 SSE 接口），客户端复用 ui/client.py 的 ApiClient。
题目取自评测集 v2：按"原题与它的口语改写放在同一组"、按题型分层，切成 4 组、每组 30 题（make_groups），
每档并发用不同的组，避免 DeepSeek 服务端前缀缓存让后面的档位显得更快。

输出：
    experiments/bench_runs/{exp_id}.jsonl  第一行元信息与汇总，之后每行一个请求（客户端计时 + 服务端耗时构成）
    experiments/bench_log.csv              每次运行一行
压测前 API 需要调高限流（bench/compose.bench.yml），否则同一 IP 的请求会被 429 挡住；大模型缓存必须关闭（默认）。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.server_log import aggregate, read_container_logs
from config import PROJECT_ROOT, get_settings
from eval.run_eval import git_commit, load_frozen_goldset, next_exp_id, percentile, write_details
from generation.llm import Price
from ui.client import ApiClient, ApiError

N_GROUPS = 4
GROUP_SEED = 11  # 分组的随机种子（D-032 登记，之后不改）
# 预热用的问题：不在 v2 中（v2 只问了五粮液 2024 年的员工人数），结果不计入
WARMUP_QUESTION = "五粮液2025年末在职员工的数量合计是多少人？"
BENCH_LOG = PROJECT_ROOT / "experiments" / "bench_log.csv"
BENCH_RUNS = PROJECT_ROOT / "experiments" / "bench_runs"
BENCH_LOG_HEADER = [
    "exp_id", "date", "git_commit", "改动说明", "mode", "concurrency", "group", "n_requests", "n_ok", "errors",
    "wall_s", "throughput_qpm", "retrieved_p50_ms", "retrieved_p90_ms", "answer_p50_ms", "answer_p90_ms", "answer_p95_ms",
    "answer_max_ms", "search_wait_p50_ms", "search_hold_p50_ms", "llm_ms_p50", "llm_calls_mean",
    "cache_hit_ratio", "cost_usd", "备注",
]  # fmt: skip


# ---------------- 题目分组 ----------------


def make_groups(items: Sequence[dict], n_groups: int = N_GROUPS, seed: int = GROUP_SEED) -> list[list[dict]]:
    """把题目切成 n_groups 组：原题与它的改写（source_id 指向原题）必须在同一组；按原题的题型分层，轮流发牌。

    同一题型内先按编号排序、再用固定种子打乱，保证可复现；组内题目顺序也用固定种子打乱，
    避免原题和它的改写紧挨着发出。
    """
    pairs: dict[str, list[dict]] = {}
    for it in items:
        pairs.setdefault(it.get("source_id") or it["id"], []).append(it)
    by_type: dict[str, list[str]] = {}
    for key, members in pairs.items():
        root = next((m for m in members if m["id"] == key), members[0])
        by_type.setdefault(root["type"], []).append(key)

    rng = random.Random(seed)
    groups: list[list[dict]] = [[] for _ in range(n_groups)]
    i = 0
    for t in sorted(by_type):
        keys = sorted(by_type[t])
        rng.shuffle(keys)
        for key in keys:
            groups[i % n_groups].extend(sorted(pairs[key], key=lambda m: m["id"]))
            i += 1
    for g in groups:
        rng.shuffle(g)
    return groups


# ---------------- 发请求 ----------------


def run_one(client: ApiClient, item: dict, t_run0: float) -> dict:
    """问一道题，记录各事件相对于发出请求的时间（毫秒）。

    start_ms：收到 start 事件（请求已通过限流和并发名额）；retrieved_ms：收到 retrieve 事件（初始检索完成）；
    answer_ms：收到答案；total_ms：事件流结束或出错。events_ms 记录每种进度事件第一次到达的时间。
    事件流开始前的错误（429、503 等）来自 ApiError。
    """
    rec: dict = {"id": item["id"], "variant": item.get("variant"), "type": item["type"],
                 "sent_at_s": round(time.perf_counter() - t_run0, 3), "status": "error"}  # fmt: skip
    t0 = time.perf_counter()

    def ms() -> float:
        return round((time.perf_counter() - t0) * 1000, 1)

    try:
        for event, data in client.stream_chat(item["question"]):
            if event == "start":
                rec.update(start_ms=ms(), trace_id=data.get("trace_id"), thread_id=data.get("thread_id"))
            elif event == "answer":
                rec.update(answer_ms=ms(), refused=data.get("refused"), format_ok=data.get("format_ok"),
                           tools=[t["tool"] for t in data.get("tool_calls", [])], steps=data.get("steps"))  # fmt: skip
            elif event == "done":
                rec["server_ms"] = data.get("latency_ms")
            elif event == "error":
                err = data.get("error", {})
                rec.update(error_code=err.get("code"), error_message=err.get("message"))
            else:
                rec.setdefault("events_ms", {}).setdefault(event, ms())
                if event == "retrieve":
                    rec["retrieved_ms"] = rec["events_ms"]["retrieve"]
        if "answer_ms" in rec and "error_code" not in rec:
            rec["status"] = "ok"
    except ApiError as e:
        rec.update(error_code=e.code, http_status=e.status, trace_id=e.trace_id)
    rec["total_ms"] = ms()
    return rec


ClientFactory = Callable[[], ApiClient]


def run_closed_loop(make_client: ClientFactory, items: Sequence[dict], concurrency: int) -> tuple[list[dict], float]:
    """闭环并发：concurrency 个线程共享一个题目队列，每个线程问完一题立刻取下一题。返回 (逐请求记录, 总墙钟秒数)。"""
    it = iter(items)
    take = threading.Lock()
    records: list[dict] = []
    t_run0 = time.perf_counter()

    def worker(w: int) -> None:
        client = make_client()
        while True:
            with take:
                item = next(it, None)
            if item is None:
                return
            rec = run_one(client, item, t_run0)
            rec["worker"] = w
            records.append(rec)  # list.append 在 CPython 中是原子操作

    with ThreadPoolExecutor(concurrency) as ex:
        list(ex.map(worker, range(concurrency)))
    return sorted(records, key=lambda r: r["sent_at_s"]), time.perf_counter() - t_run0


def run_burst(make_client: ClientFactory, items: Sequence[dict], size: int, rounds: int,
              verify: int = 0) -> tuple[list[dict], float]:  # fmt: skip
    """突发：每轮 size 个线程在 Barrier 处对齐后同时发请求，等本轮全部结束再开始下一轮；题目循环使用。

    verify > 0 时，最后再同时发 verify 个请求（round="verify"）：此时之前的请求都已结束，
    若并发名额没有泄漏，这一轮应当全部被接受。
    """
    plan = [(r, size) for r in range(rounds)] + ([("verify", verify)] if verify else [])
    records: list[dict] = []
    k = 0
    t_run0 = time.perf_counter()
    for rnd, n in plan:
        batch = [items[(k + j) % len(items)] for j in range(n)]
        k += n
        barrier = threading.Barrier(n)

        def fire(item: dict, rnd=rnd, barrier=barrier) -> dict:
            client = make_client()
            barrier.wait()
            rec = run_one(client, item, t_run0)
            rec["round"] = rnd
            return rec

        with ThreadPoolExecutor(n) as ex:
            records.extend(ex.map(fire, batch))
    return records, time.perf_counter() - t_run0


# ---------------- 汇总 ----------------


def _p(values: list[float], q: float) -> float | None:
    v = percentile(values, q)
    return None if v is None else round(v, 1)


def attach_server_stats(records: list[dict], server: dict[str, dict]) -> int:
    """把服务端统计并入每条记录（按 trace_id）。返回成功匹配的请求数。"""
    n = 0
    for r in records:
        s = server.get(r.get("trace_id") or "")
        if s and r["status"] == "ok":
            r["server"] = s
            n += 1
    return n


def summarize(records: list[dict], wall_s: float) -> dict:
    """一次运行的汇总。延迟分位数只统计成功的请求；被拒绝的请求单独统计拒绝耗时。"""
    ok = [r for r in records if r["status"] == "ok"]
    errors = Counter(r.get("error_code") or "unknown" for r in records if r["status"] != "ok")
    srv = [r["server"] for r in ok if "server" in r]
    prompt = sum(s["prompt_tokens"] for s in srv)
    rejected = [r["total_ms"] for r in records if r.get("http_status") in (429, 503)]
    return {
        "n_requests": len(records),
        "n_ok": len(ok),
        "errors": dict(sorted(errors.items())),
        "wall_s": round(wall_s, 1),
        "throughput_qpm": round(len(ok) / wall_s * 60, 2) if wall_s > 0 else None,
        "retrieved_p50_ms": _p([r["retrieved_ms"] for r in ok if "retrieved_ms" in r], 50),
        "retrieved_p90_ms": _p([r["retrieved_ms"] for r in ok if "retrieved_ms" in r], 90),
        "answer_p50_ms": _p([r["answer_ms"] for r in ok], 50),
        "answer_p90_ms": _p([r["answer_ms"] for r in ok], 90),
        "answer_p95_ms": _p([r["answer_ms"] for r in ok], 95),
        "answer_max_ms": _p([r["answer_ms"] for r in ok], 100),
        "n_server_matched": len(srv),
        "search_wait_p50_ms": _p([s["search_wait_ms"] for s in srv], 50),
        "search_wait_p90_ms": _p([s["search_wait_ms"] for s in srv], 90),
        "search_hold_p50_ms": _p([s["search_hold_ms"] for s in srv], 50),
        "llm_ms_p50": _p([s["llm_ms"] for s in srv], 50),
        "llm_calls_mean": round(sum(s["n_llm"] for s in srv) / len(srv), 2) if srv else None,
        "cache_hit_ratio": round(sum(s["cache_hit_tokens"] for s in srv) / prompt, 4) if prompt else None,
        "cost_usd": round(sum(s.get("cost_usd", 0) for s in srv), 4) if srv else None,
        "rejected_p50_ms": _p(rejected, 50),
        "rejected_max_ms": _p(rejected, 100),
        "by_round": _by_round(records),
    }


def _by_round(records: list[dict]) -> dict | None:
    """突发模式：每轮接受与拒绝的请求数。闭环模式返回 None。"""
    if not any("round" in r for r in records):
        return None
    out: dict[str, dict] = {}
    for r in records:
        d = out.setdefault(str(r["round"]), {"accepted": 0, "rejected_503": 0, "other": 0})
        if r.get("http_status") == 503:
            d["rejected_503"] += 1
        elif "start_ms" in r:
            d["accepted"] += 1
        else:
            d["other"] += 1
    return out


def append_bench_log(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BENCH_LOG_HEADER)
        if new:
            w.writeheader()
        w.writerow({k: "" if row.get(k) is None else row[k] for k in BENCH_LOG_HEADER})


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def print_summary(exp_id: str, s: dict) -> None:
    print(f"[{exp_id}] 请求 {s['n_requests']}，成功 {s['n_ok']}，错误 {s['errors'] or '无'}，"
          f"墙钟 {s['wall_s']}s，吞吐 {_fmt(s['throughput_qpm'])} 题/分钟")  # fmt: skip
    print(f"  检索完成 p50/p90 {_fmt(s['retrieved_p50_ms'])}/{_fmt(s['retrieved_p90_ms'])}ms；答案 p50/p90/p95/max "
          f"{_fmt(s['answer_p50_ms'])}/{_fmt(s['answer_p90_ms'])}/{_fmt(s['answer_p95_ms'])}/{_fmt(s['answer_max_ms'])}ms")
    print(f"  服务端（{s['n_server_matched']} 个请求）：检索排队 p50 {_fmt(s['search_wait_p50_ms'])}ms "
          f"p90 {_fmt(s['search_wait_p90_ms'])}ms，检索执行 p50 {_fmt(s['search_hold_p50_ms'])}ms，"
          f"大模型 p50 {_fmt(s['llm_ms_p50'])}ms（平均 {_fmt(s['llm_calls_mean'])} 次），"
          f"缓存命中 {_fmt(s['cache_hit_ratio'])}，费用 ${_fmt(s['cost_usd'])}")
    if s["by_round"]:
        print(f"  各轮：{s['by_round']}；拒绝耗时 p50 {_fmt(s['rejected_p50_ms'])}ms max {_fmt(s['rejected_max_ms'])}ms")


# ---------------- 命令行 ----------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="阶段 11 轻量压测（D-032）")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--group", type=int, required=True, choices=range(N_GROUPS), help="使用第几组题目（0~3）")
    p.add_argument("--concurrency", type=int, default=1, help="闭环并发数")
    p.add_argument("--burst", type=int, default=0, help="突发模式：每轮同时发出的请求数（>0 时启用）")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--verify", type=int, default=0, help="突发结束后再同时发几个请求，检查并发名额是否泄漏")
    p.add_argument("--limit", type=int, default=0, help="只用本组前 N 题（冒烟测试用）")
    p.add_argument("--warmup", type=int, default=1,
                   help="正式计时前先发几个预热请求（不计入结果）：容器启动后第一次检索有约 2 秒的冷启动")
    p.add_argument("--container", default="deploy-api-1", help="api 容器名，用于读取服务端日志")
    p.add_argument("--desc", default="", help="改动说明，写入 bench_log.csv")
    p.add_argument("--no-log", action="store_true", help="不写 bench_log.csv（冒烟测试），明细写到 _nolog_ 文件")
    args = p.parse_args(argv)

    settings = get_settings()
    frozen = settings.eval.goldsets["v2"]
    items = load_frozen_goldset(settings.eval.goldset_file("v2"), frozen.sha256)
    group = make_groups(items)[args.group]
    if args.limit:
        group = group[: args.limit]

    def make_client() -> ApiClient:
        return ApiClient(args.base_url, timeout_s=settings.ui.timeout_s)

    ready = make_client().ready()
    if ready.get("status") != "ok":
        p.error(f"API 未就绪：{ready}")

    for _ in range(args.warmup):
        run_one(make_client(), {"id": "warmup", "type": "warmup", "question": WARMUP_QUESTION}, time.perf_counter())

    since = dt.datetime.now(dt.timezone.utc).isoformat()
    if args.burst:
        mode, conc = "burst", args.burst
        records, wall = run_burst(make_client, group, args.burst, args.rounds, args.verify)
    else:
        mode, conc = "closed_loop", args.concurrency
        records, wall = run_closed_loop(make_client, group, args.concurrency)

    time.sleep(1)  # 等最后几条日志写出
    price = Price(settings.llm.price_input_cache_hit, settings.llm.price_input_cache_miss, settings.llm.price_output)
    try:
        server = aggregate(read_container_logs(args.container, since), price)
    except Exception as e:  # noqa: BLE001  读不到日志时只缺服务端统计，客户端结果照常记录
        print(f"警告：读取容器 {args.container} 的日志失败（{type(e).__name__}），没有服务端耗时构成")
        server = {}
    attach_server_stats(records, server)
    summary = summarize(records, wall)

    exp_id = "_nolog_" + dt.datetime.now().strftime("%H%M%S") if args.no_log else next_exp_id(BENCH_LOG, "B")
    commit = git_commit()
    meta = {"exp_id": exp_id, "date": dt.date.today().isoformat(), "git_commit": commit, "desc": args.desc,
            "mode": mode, "concurrency": conc, "group": args.group, "rounds": args.rounds if args.burst else None,
            "verify": args.verify or None, "warmup": args.warmup, "base_url": args.base_url, "started_at": since,
            "goldset_version": "v2", "goldset_sha256": frozen.sha256, "group_seed": GROUP_SEED,
            "question_ids": [f"{it['id']}" for it in group], "llm_model": settings.llm.model, "summary": summary}  # fmt: skip
    write_details(BENCH_RUNS / f"{exp_id}.jsonl", meta, records)
    print_summary(exp_id, summary)
    if not args.no_log:
        row = {"exp_id": exp_id, "date": meta["date"], "git_commit": commit, "改动说明": args.desc, "mode": mode,
               "concurrency": conc, "group": args.group,
               **{k: v for k, v in summary.items() if k in BENCH_LOG_HEADER},
               "errors": ";".join(f"{k}={v}" for k, v in summary["errors"].items()),
               "备注": f"明细 experiments/bench_runs/{exp_id}.jsonl"}  # fmt: skip
        append_bench_log(BENCH_LOG, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
