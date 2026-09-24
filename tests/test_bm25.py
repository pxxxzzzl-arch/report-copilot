"""retrieval/bm25.py 的单元测试：打分用手算值核对。"""

import math
from dataclasses import replace

import pytest

from retrieval.bm25 import BM25Index, BM25Retriever, tokenize
from retrieval.query import QueryFilter
from tests.test_index_dense import _chunk

# 语料：d0=[a,b]，d1=[a,a,c]，d2=[c]；N=3，文档长度 2/3/1，avgdl=2；k1=1.5，b=0.75
DOCS = [["a", "b"], ["a", "a", "c"], ["c"]]
IDX = BM25Index(DOCS, k1=1.5, b=0.75)
IDF_A = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))  # df(a)=2 -> ln(1.6) ≈ 0.4700


def test_idf() -> None:
    assert IDX.idf("a") == pytest.approx(IDF_A)
    assert IDX.idf("b") == pytest.approx(math.log(1 + 2.5 / 1.5))  # df=1 -> ln(2.6667) ≈ 0.9808
    assert IDX.idf("zzz") == pytest.approx(math.log(1 + 3.5 / 0.5))  # df=0 也有定义，但检索时会被跳过


def test_scores_hand_calculated() -> None:
    s = IDX.scores(["a"])
    # d0：tf=1，dl=2 -> norm = 1.5*(0.25 + 0.75*2/2) = 1.5 -> 1*2.5/(1+1.5) = 1.0
    assert s[0] == pytest.approx(IDF_A * 1.0)
    # d1：tf=2，dl=3 -> norm = 1.5*(0.25 + 0.75*3/2) = 2.0625 -> 2*2.5/(2+2.0625) ≈ 1.2308
    assert s[1] == pytest.approx(IDF_A * 5 / 4.0625)
    assert 2 not in s  # d2 不含 a
    assert s[1] > s[0]  # 词频更高的得分更高（但不是翻倍：词频饱和）


def test_length_normalization() -> None:
    s = IDX.scores(["c"])
    # 都是 tf=1：短文档 d2（dl=1）得分高于长文档 d1（dl=3）
    assert s[2] > s[1]
    # b=0 时不做长度归一化，两者相同
    s0 = BM25Index(DOCS, k1=1.5, b=0.0).scores(["c"])
    assert s0[1] == pytest.approx(s0[2])


def test_query_terms_deduplicated_and_unknown_ignored() -> None:
    assert IDX.scores(["a", "a"]) == IDX.scores(["a"])
    assert IDX.scores(["zzz"]) == {}


def test_tokenize_drops_punctuation_and_stopwords() -> None:
    toks = tokenize("贵州茅台２０２４年研发投入是多少？")
    assert "？" not in toks and "?" not in toks and "多少" not in toks and "是" not in toks
    assert "研发" in toks and "2024" in toks  # 全角数字转半角
    assert "归属于" in tokenize("归属于上市公司股东的净利润")  # 搜索模式保留长词


def test_retriever_search_with_filter() -> None:
    chunks = [
        replace(_chunk(0, "研发投入合计 100"), chunk_id="A24", stock_code="600519", year=2024),
        replace(_chunk(1, "研发投入合计 200"), chunk_id="A25", stock_code="600519", year=2025),
        replace(_chunk(2, "在职员工的数量合计"), chunk_id="B24", stock_code="000858", year=2024),
    ]
    r = BM25Retriever(chunks, k1=1.5, b=0.75)
    top = r.search("研发投入", 10)
    assert {c.chunk_id for c in top} == {"A24", "A25"}  # 不含"研发投入"的 B24 没有得分，不返回
    assert all("bm25_score" in c.extra for c in top)
    assert [c.chunk_id for c in r.search("研发投入", 10, QueryFilter(("600519",), (2025,)))] == ["A25"]
    assert len(r.search("研发投入", 1)) == 1
