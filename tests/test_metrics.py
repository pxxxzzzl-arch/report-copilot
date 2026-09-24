"""eval/metrics.py 的单元测试：每个指标都有写明手算过程的用例。"""

import math
from dataclasses import dataclass

import pytest

from eval.metrics import (
    GoldEvidence,
    chunk_hits_gold,
    hit_at_k,
    hit_matrix,
    mean_scores,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_question,
)


@dataclass
class C:
    """测试用的最小 chunk。"""

    doc_id: str
    page_start: int
    page_end: int
    text: str
    chunk_id: str = "c"


def ev(doc_id: str, page: int, span: str, alt: list[tuple[str, int]] = ()) -> GoldEvidence:
    return GoldEvidence.from_dict(
        {"doc_id": doc_id, "page": page, "span": span, "alt": [{"doc_id": d, "page": p} for d, p in alt]}
    )


# ---------------- 命中判定 ----------------


def test_hit_requires_doc_page_and_span() -> None:
    g = ev("A_2024", 5, "1,234.56")
    assert chunk_hits_gold(C("A_2024", 5, 5, "收入 1,234.56 元"), g)
    assert not chunk_hits_gold(C("B_2024", 5, 5, "收入 1,234.56 元"), g)  # 文档不同
    assert not chunk_hits_gold(C("A_2024", 6, 6, "收入 1,234.56 元"), g)  # 页码不同
    assert not chunk_hits_gold(C("A_2024", 5, 5, "收入 1,234.57 元"), g)  # 文本不含 span


def test_hit_page_range_inclusive() -> None:
    g = ev("A_2024", 5, "abcd")
    assert chunk_hits_gold(C("A_2024", 4, 5, "abcd"), g)  # 跨页 chunk，右端点
    assert chunk_hits_gold(C("A_2024", 5, 7, "abcd"), g)  # 左端点
    assert not chunk_hits_gold(C("A_2024", 6, 7, "abcd"), g)


def test_hit_after_normalization() -> None:
    # chunk 中数字被断行、带空格、全角：归一化后仍能命中
    g = ev("A_2024", 5, "1,234,567.89")
    assert chunk_hits_gold(C("A_2024", 5, 5, "营业收入 １,234,\n567.89"), g)


def test_hit_via_alt_location() -> None:
    g = ev("A_2024", 5, "abcd", alt=[("A_2024", 90), ("A_2025", 7)])
    assert chunk_hits_gold(C("A_2024", 90, 90, "xxabcdxx"), g)
    assert chunk_hits_gold(C("A_2025", 7, 7, "abcd"), g)
    assert not chunk_hits_gold(C("A_2025", 8, 8, "abcd"), g)


def test_hit_matrix() -> None:
    golds = [ev("A", 1, "aaaa"), ev("A", 2, "bbbb")]
    chunks = [C("A", 3, 3, "zzzz"), C("A", 1, 2, "aaaa bbbb"), C("A", 2, 2, "bbbb")]
    assert hit_matrix(chunks, golds) == [frozenset(), frozenset({0, 1}), frozenset({1})]


# ---------------- 各指标：手算用例 ----------------
# 通用场景：2 条 gold，检索结果命中矩阵为 [无, {0}, 无, {0}, {1}]
H = [frozenset(), frozenset({0}), frozenset(), frozenset({0}), frozenset({1})]


def test_hit_at_k() -> None:
    assert hit_at_k(H, 1) == 0  # 第 1 名没命中
    assert hit_at_k(H, 2) == 1  # 第 2 名命中 gold0
    assert hit_at_k([], 5) == 0  # 空结果


def test_recall_at_k() -> None:
    assert recall_at_k(H, 2, 1) == 0  # 覆盖 {} -> 0/2
    assert recall_at_k(H, 2, 4) == 0.5  # 覆盖 {0}（第 2、4 名重复命中只算一次）-> 1/2
    assert recall_at_k(H, 2, 5) == 1.0  # 覆盖 {0,1} -> 2/2
    assert recall_at_k([], 2, 5) == 0


def test_precision_at_k() -> None:
    assert precision_at_k(H, 5) == pytest.approx(3 / 5)  # 第 2、4、5 名命中
    assert precision_at_k(H, 2) == pytest.approx(1 / 2)
    # 只返回 2 个结果但 k=10：分母仍是 10 -> 1/10
    assert precision_at_k(H[:2], 10) == pytest.approx(1 / 10)


def test_mrr_at_k() -> None:
    assert mrr_at_k(H, 10) == pytest.approx(1 / 2)  # 第一个命中在第 2 名
    assert mrr_at_k(H, 1) == 0  # 前 1 名没有命中
    assert mrr_at_k([frozenset({0})], 10) == 1.0


def test_ndcg_basic() -> None:
    # 增益（新覆盖条数）：第 2 名覆盖 gold0 得 1；第 4 名重复 gold0 得 0；第 5 名覆盖 gold1 得 1
    # DCG@5  = 1/log2(3) + 1/log2(6)
    # IDCG@5 = 1/log2(2) + 1/log2(3)（2 条 gold，理想情况是排在第 1、2 名）
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(H, 2, 5) == pytest.approx(dcg / idcg)  # (0.6309 + 0.3869) / 1.6309 ≈ 0.6241
    assert ndcg_at_k(H, 2, 5) == pytest.approx(0.6241, abs=1e-4)


def test_ndcg_perfect_order() -> None:
    assert ndcg_at_k([frozenset({0}), frozenset({1})], 2, 10) == pytest.approx(1.0)


def test_ndcg_duplicates_not_rewarded() -> None:
    # 同一条 gold 被前 3 名重复命中，另外 2 条都没找到：
    # DCG = 1；IDCG@3 = 1 + 1/log2(3) + 1/log2(4) ≈ 2.1309 -> ≈ 0.4693
    hits = [frozenset({0})] * 3
    assert ndcg_at_k(hits, 3, 3) == pytest.approx(1 / (1 + 1 / math.log2(3) + 0.5))


def test_ndcg_multiple_gold_in_one_chunk_capped() -> None:
    # 两条 gold 在同一页，第 1 名的 chunk 同时覆盖：DCG = 2 > IDCG = 1 + 1/log2(3) ≈ 1.63，截断为 1
    assert ndcg_at_k([frozenset({0, 1})], 2, 5) == 1.0


def test_ndcg_k_smaller_than_n_gold() -> None:
    # 3 条 gold、k=1：IDCG@1 = 1；第 1 名覆盖 1 条 -> 1.0
    assert ndcg_at_k([frozenset({2}), frozenset({0})], 3, 1) == 1.0


def test_ndcg_empty() -> None:
    assert ndcg_at_k([], 2, 5) == 0


# ---------------- 汇总 ----------------


def test_score_question_keys_and_values() -> None:
    s = score_question(H, 2, [1, 5])
    assert set(s) == {f"{m}@{k}" for m in ("hit", "recall", "precision", "mrr", "ndcg") for k in (1, 5)}
    assert s["hit@1"] == 0 and s["hit@5"] == 1
    assert s["recall@5"] == 1.0 and s["mrr@5"] == 0.5


def test_score_question_rejects_no_gold() -> None:
    with pytest.raises(ValueError):
        score_question([], 0, [5])


def test_mean_scores() -> None:
    rows = [{"hit@5": 1.0, "recall@5": 0.5}, {"hit@5": 0.0, "recall@5": 1.0}]
    assert mean_scores(rows) == {"hit@5": 0.5, "recall@5": 0.75}
    assert mean_scores([]) == {}
