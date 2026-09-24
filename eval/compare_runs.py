"""对比两次实验：差异是真实的改进，还是 52 道题上的随机波动？

读取两次实验的逐题明细（experiments/runs/{exp_id}.jsonl），对每个指标给出：
    均值差       new - base
    95% 置信区间  配对 bootstrap：有放回地重抽题目 n_boot 次，每次算均值差，取 2.5% 和 97.5% 分位数
                 "配对"指同一道题在两次实验中的得分一起被抽中，这样题目本身的难易程度会被抵消
    符号检验      变好的题数 vs 变差的题数（平局不计），双侧精确二项检验的 p 值
以及具体哪些题变好了、哪些题变差了（便于回头看 Bad Case）。

判定口径（docs/decisions.md D-015）：置信区间不跨 0 才称为"确定有效"，其余如实写"方向为正但不显著"。

用法：
    uv run python -m eval.compare_runs E001 E002
    uv run python -m eval.compare_runs E001 E002 --metrics hit@5 recall@10
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from config import get_settings

DEFAULT_METRICS = ("hit@5", "recall@5", "mrr@10", "ndcg@10")


@dataclass
class MetricComparison:
    metric: str
    base_mean: float
    new_mean: float
    diff: float
    ci_low: float
    ci_high: float
    n_better: int
    n_worse: int
    sign_p: float
    better_ids: list[str]
    worse_ids: list[str]

    @property
    def verdict(self) -> str:
        if self.ci_low > 0:
            return "确定提升"
        if self.ci_high < 0:
            return "确定下降"
        return "不显著"


def load_scores(path: Path) -> tuple[dict, dict[str, dict[str, float]]]:
    """读取逐题明细，返回 (元信息, {题目 id: 各项得分})。"""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return lines[0]["meta"], {r["id"]: r["scores"] for r in lines[1:]}


def paired_bootstrap_ci(
    base: Sequence[float], new: Sequence[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """配对 bootstrap 均值差 (new - base) 的 (1 - alpha) 置信区间（百分位法）。固定种子，结果可复现。"""
    diffs = [n - b for b, n in zip(base, new, strict=True)]
    rng = random.Random(seed)
    size = len(diffs)
    means = sorted(sum(diffs[rng.randrange(size)] for _ in range(size)) / size for _ in range(n_boot))
    lo = means[int(math.floor(alpha / 2 * n_boot))]
    hi = means[min(int(math.ceil((1 - alpha / 2) * n_boot)) - 1, n_boot - 1)]
    return lo, hi


def sign_test_p(n_better: int, n_worse: int) -> float:
    """双侧精确符号检验：零假设下每道非平局题"变好"的概率为 1/2。

    p = 2 * P(X <= min(变好, 变差))，X ~ Binomial(变好 + 变差, 1/2)，上限截断为 1。
    """
    n = n_better + n_worse
    if n == 0:
        return 1.0
    k = min(n_better, n_worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(
    base: dict[str, dict[str, float]], new: dict[str, dict[str, float]], metrics: Sequence[str], **boot_kw
) -> list[MetricComparison]:
    """对两次实验共有的题目逐项比较。两次的题目集合必须一致（同一评测集版本）。"""
    if set(base) != set(new):
        raise ValueError("两次实验的题目集合不一致，无法配对比较（检查评测集版本）")
    ids = sorted(base)
    out = []
    for m in metrics:
        b = [base[q][m] for q in ids]
        n = [new[q][m] for q in ids]
        lo, hi = paired_bootstrap_ci(b, n, **boot_kw)
        better = [q for q, x, y in zip(ids, b, n) if y > x]
        worse = [q for q, x, y in zip(ids, b, n) if y < x]
        out.append(
            MetricComparison(
                metric=m, base_mean=sum(b) / len(b), new_mean=sum(n) / len(n), diff=(sum(n) - sum(b)) / len(b),
                ci_low=lo, ci_high=hi, n_better=len(better), n_worse=len(worse),
                sign_p=sign_test_p(len(better), len(worse)), better_ids=better, worse_ids=worse,
            )  # fmt: skip
        )
    return out


def format_report(base_id: str, new_id: str, rows: list[MetricComparison]) -> str:
    lines = [
        f"{base_id} -> {new_id}",
        f"{'指标':<12}{'base':>8}{'new':>8}{'差值':>9}{'95% CI':>20}{'变好/变差':>11}{'符号检验p':>11}  结论",
    ]
    for r in rows:
        ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        lines.append(
            f"{r.metric:<12}{r.base_mean:>8.4f}{r.new_mean:>8.4f}{r.diff:>+9.4f}{ci:>20}"
            f"{f'{r.n_better}/{r.n_worse}':>11}{r.sign_p:>11.4f}  {r.verdict}"
        )
    for r in rows:
        lines.append(f"\n[{r.metric}] 变好：{' '.join(r.better_ids) or '无'}")
        lines.append(f"[{r.metric}] 变差：{' '.join(r.worse_ids) or '无'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="对比两次检索实验（配对 bootstrap + 符号检验）")
    p.add_argument("base", help="基线实验 id，如 E001")
    p.add_argument("new", help="新实验 id，如 E002")
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    args = p.parse_args(argv)

    runs_dir = get_settings().paths.experiments_dir / "runs"
    base_meta, base = load_scores(runs_dir / f"{args.base}.jsonl")
    new_meta, new = load_scores(runs_dir / f"{args.new}.jsonl")
    if base_meta.get("goldset_sha256") != new_meta.get("goldset_sha256"):
        raise SystemExit("两次实验使用的评测集不同（sha256 不一致），不能比较")
    print(format_report(args.base, args.new, compare(base, new, args.metrics)))


if __name__ == "__main__":
    main()
