"""eval/compare_runs.py 的单元测试：统计量用手算值核对。"""

import json
from pathlib import Path

import pytest

from eval.compare_runs import compare, gen_scores, load_scores, paired_bootstrap_ci, sign_test_p


@pytest.mark.parametrize(
    ("better", "worse", "expected"),
    [
        (0, 0, 1.0),  # 没有非平局题
        (5, 0, 2 / 32),  # n=5，P(X<=0) = 1/32，双侧 = 1/16 = 0.0625
        (6, 1, 2 * (1 + 7) / 128),  # n=7，P(X<=1) = (C(7,0)+C(7,1))/2^7 = 8/128，双侧 = 0.125
        (3, 3, 1.0),  # 完全对称，2*P(X<=3) = 2*(1+6+15+20)/64 > 1，截断为 1
        (1, 6, 2 * (1 + 7) / 128),  # 与 (6,1) 对称
    ],
)
def test_sign_test_p(better: int, worse: int, expected: float) -> None:
    assert sign_test_p(better, worse) == pytest.approx(expected)


def test_bootstrap_identical_runs_zero_interval() -> None:
    a = [1.0, 0.0, 1.0, 0.5]
    assert paired_bootstrap_ci(a, a, n_boot=500) == (0.0, 0.0)


def test_bootstrap_constant_shift() -> None:
    # 每题都 +0.25：任何重抽样的均值差都是 0.25
    base = [0.0, 0.25, 0.5, 0.75]
    new = [x + 0.25 for x in base]
    lo, hi = paired_bootstrap_ci(base, new, n_boot=500)
    assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)


def test_bootstrap_interval_contains_mean_and_is_reproducible() -> None:
    base = [0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
    new = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]  # 均值差 = 3/10 = 0.3
    ci1 = paired_bootstrap_ci(base, new, n_boot=2000, seed=1)
    assert ci1 == paired_bootstrap_ci(base, new, n_boot=2000, seed=1)  # 固定种子可复现
    assert ci1[0] <= 0.3 <= ci1[1]
    assert ci1[0] >= 0  # 只变好不变差：区间下界不可能小于 0


def test_compare_lists_changed_questions() -> None:
    base = {"q1": {"hit@5": 0.0}, "q2": {"hit@5": 1.0}, "q3": {"hit@5": 1.0}}
    new = {"q1": {"hit@5": 1.0}, "q2": {"hit@5": 0.0}, "q3": {"hit@5": 1.0}}
    (r,) = compare(base, new, ["hit@5"], n_boot=200)
    assert (r.base_mean, r.new_mean, r.diff) == (pytest.approx(2 / 3), pytest.approx(2 / 3), 0.0)
    assert r.better_ids == ["q1"] and r.worse_ids == ["q2"]
    assert r.sign_p == 1.0 and r.verdict == "不显著"


def test_compare_rejects_mismatched_questions() -> None:
    with pytest.raises(ValueError, match="不一致"):
        compare({"q1": {"hit@5": 1}}, {"q2": {"hit@5": 1}}, ["hit@5"])


def test_load_scores(tmp_path: Path) -> None:
    path = tmp_path / "E001.jsonl"
    rows = [{"meta": {"exp_id": "E001"}}, {"id": "q1", "scores": {"hit@5": 1.0}}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    meta, scores = load_scores(path)
    assert meta["exp_id"] == "E001" and scores == {"q1": {"hit@5": 1.0}}


def test_load_scores_subset(tmp_path: Path) -> None:
    path = tmp_path / "E009.jsonl"
    rows = [
        {"meta": {}},
        {"id": "q001", "variant": "original", "scores": {"hit@5": 1.0}},
        {"id": "q101", "variant": "paraphrase", "scores": {"hit@5": 0.0}},
        {"id": "q002", "scores": {"hit@5": 1.0}},  # 没有 variant 字段（v1 的明细）视为 original
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert set(load_scores(path)[1]) == {"q001", "q101", "q002"}
    assert set(load_scores(path, "original")[1]) == {"q001", "q002"}
    assert set(load_scores(path, "paraphrase")[1]) == {"q101"}



def test_gen_scores() -> None:
    answerable = {"judge": {"correctness": 1}, "metrics": {"cite_recall": 0.5, "refused_ok": True}}
    oos = {"judge": {"correctness": 2}, "metrics": {"refused_ok": False}}
    assert gen_scores(answerable) == {"judge_correct": 0.5, "cite_recall": 0.5, "refused_ok": 1.0}
    assert gen_scores(oos) == {"judge_correct": 1.0, "cite_recall": 1.0, "refused_ok": 0.0}
