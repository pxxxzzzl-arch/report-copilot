"""retrieval/query.py、fusion.py、search.py 以及 dense 过滤的单元测试。"""

from dataclasses import replace

import pytest
from qdrant_client import QdrantClient

from config import Settings
from ingest.download import ManifestRow
from ingest.index import ensure_collection, index_chunks
from retrieval.dense import DenseRetriever, to_qdrant_filter
from retrieval.fusion import rrf_fuse, rrf_scores
from retrieval.query import QueryFilter, QueryParser
from retrieval.search import SearchPipeline
from tests.test_index_dense import DIM, KeywordEmbedder, _chunk

MANIFEST = {
    d: ManifestRow(d, name, d[:6], int(d[-4:]), "", 1, "", "", "", "")
    for d, name in [("600519_2024", "贵州茅台"), ("600519_2025", "贵州茅台"), ("000858_2024", "五粮液"), ("300750_2025", "宁德时代")]
}
PARSER = QueryParser(MANIFEST, {"茅台": "600519", "CATL": "300750"})


# ---------------- 查询理解 ----------------


@pytest.mark.parametrize(
    ("query", "codes", "years"),
    [
        ("贵州茅台2024年研发投入合计是多少？", ("600519",), (2024,)),
        ("茅台 2025 年的营业收入", ("600519",), (2025,)),  # 别名 + 年份中间有空格
        ("2024年贵州茅台和五粮液谁的研发投入更高？", ("000858", "600519"), (2024,)),  # 多家公司
        ("五粮液2025年营业收入比2024年减少了多少？", ("000858",), (2024, 2025)),  # 多个年份
        ("CATL 的注册地址在哪里？", ("300750",), ()),  # 英文别名，无年份
        ("600519 的董事长是谁？", ("600519",), ()),  # 股票代码
        ("招商银行2025年的净息差是多少？", (), (2025,)),  # 语料外公司：识别不到公司
        ("派发现金红利1,256股，合计2024元", (), ()),  # 四位数字后没有"年"，不算年份
        ("2024年度利润分配预案", (), (2024,)),  # "年度"也算
    ],
)
def test_query_parser(query: str, codes: tuple, years: tuple) -> None:
    assert PARSER.parse(query) == QueryFilter(codes, years)


def test_query_filter_matches() -> None:
    f = QueryFilter(("600519",), (2024,))
    assert f.matches("600519", 2024)
    assert not f.matches("600519", 2025)
    assert not f.matches("000858", 2024)
    assert QueryFilter((), (2024,)).matches("000858", 2024)  # 公司不限
    assert QueryFilter().is_empty and QueryFilter().matches("x", 1)


def test_to_qdrant_filter() -> None:
    assert to_qdrant_filter(None) is None
    assert to_qdrant_filter(QueryFilter()) is None
    f = to_qdrant_filter(QueryFilter(("600519", "000858"), (2024,)))
    assert [c.key for c in f.must] == ["stock_code", "year"]
    assert f.must[0].match.any == ["600519", "000858"]


# ---------------- RRF（手算） ----------------


def test_rrf_scores_hand_calculated() -> None:
    # k=60。a 在第 1 路排第 1、第 2 路排第 2：1/61 + 1/62；b：1/62 + 1/61；c 只在第 1 路排第 3：1/63
    s = rrf_scores([["a", "b", "c"], ["b", "a"]], k=60)
    assert s["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["c"] == pytest.approx(1 / 63)


def test_rrf_fuse_order_and_tiebreak() -> None:
    x, y, z = (replace(_chunk(i, t), chunk_id=cid) for i, t, cid in [(0, "x", "x"), (1, "y", "y"), (2, "z", "z")])
    # x：1/61 + 1/63 ≈ 0.03227；y：1/62 + 1/61 ≈ 0.03252；z：1/63
    fused = rrf_fuse([[x, y, z], [y, z, x]], k=60)
    assert [c.chunk_id for c in fused] == ["y", "x", "z"]  # 两路都靠前的 y 胜出
    assert fused[0].extra["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    # 同分：两个 chunk 各自只在一路排第 1，分数相同 -> 比最好排名（相同）-> 比 chunk_id
    assert [c.chunk_id for c in rrf_fuse([[y], [x]], k=60)] == ["x", "y"]


def test_rrf_k_changes_ranking() -> None:
    # a 只在第 1 路排第 1；b 在两路都排第 4
    #   k=1 ：a = 1/2 = 0.5，b = 1/5 + 1/5 = 0.4 -> a 胜（小 k 放大头部排名的优势）
    #   k=60：a = 1/61 ≈ 0.0164，b = 2/64 ≈ 0.0313 -> b 胜（大 k 更看重"多路都召回"）
    a, b, x, y, p, q, r = (replace(_chunk(i, t), chunk_id=t) for i, t in enumerate("abxypqr"))
    lists = [[a, x, y, b], [p, q, r, b]]
    assert rrf_fuse(lists, k=1)[0].chunk_id == "a"  # a 与 p 同为 0.5，同分按 chunk_id，a 在前
    assert rrf_fuse(lists, k=60)[0].chunk_id == "b"


# ---------------- 带过滤的稠密检索 + 流水线 ----------------

# 两份年报各有一个"研发投入"chunk：没有过滤时它们向量相同、难以区分，年份过滤后只剩对的那个
FILTER_CHUNKS = [
    replace(_chunk(0, "研发投入合计 100"), chunk_id="A24", doc_id="600519_2024", stock_code="600519", year=2024),
    replace(_chunk(1, "研发投入合计 200"), chunk_id="A25", doc_id="600519_2025", stock_code="600519", year=2025),
    replace(_chunk(2, "研发投入合计 300"), chunk_id="B24", doc_id="000858_2024", stock_code="000858", year=2024),
]


@pytest.fixture
def filtered_env():
    client = QdrantClient(":memory:")
    emb = KeywordEmbedder()
    ensure_collection(client, "c", DIM)
    index_chunks(client, "c", FILTER_CHUNKS, emb, batch=10)
    return client, emb


def test_dense_search_with_filter(filtered_env) -> None:
    client, emb = filtered_env
    r = DenseRetriever(client, "c", emb, Settings(_env_file=None))
    assert len(r.search("研发投入", 10)) == 3  # 不过滤
    got = r.search("研发投入", 10, QueryFilter(("600519",), (2025,)))
    assert [c.chunk_id for c in got] == ["A25"]
    got = r.search("研发投入", 10, QueryFilter((), (2024,)))
    assert sorted(c.chunk_id for c in got) == ["A24", "B24"]


def test_pipeline_single_source_equals_plain_dense(filtered_env) -> None:
    client, emb = filtered_env
    settings = Settings(_env_file=None)
    dense = DenseRetriever(client, "c", emb, settings)
    pipe = SearchPipeline(settings, {"dense": dense})
    q = "研发投入"
    assert [c.chunk_id for c in pipe.search(q, 2)] == [c.chunk_id for c in dense.search(q, 2)]
    assert pipe.config["retriever"] == "dense" and pipe.config["fusion"] == "" and pipe.config["top_n"] == ""


def test_pipeline_with_filter(filtered_env) -> None:
    client, emb = filtered_env
    settings = Settings(_env_file=None)
    pipe = SearchPipeline(settings, {"dense": DenseRetriever(client, "c", emb, settings)}, parser=PARSER,
                          source_labels={"dense": "dense(fake)"})  # fmt: skip
    got = pipe.search("贵州茅台2025年研发投入合计是多少？", 10)
    assert [c.chunk_id for c in got] == ["A25"]
    assert pipe.config["retriever"] == "filter+dense(fake)"


class ListSource:
    """按固定顺序返回的假召回通道，同时记录收到的参数。"""

    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def search(self, query, top_k, flt=None):
        self.calls.append((top_k, flt))
        return self.chunks[:top_k]


def test_pipeline_multi_source_fuses_with_top_n() -> None:
    a, b, c = (replace(_chunk(i, t), chunk_id=t) for i, t in enumerate("abc"))
    s1, s2 = ListSource([a, b]), ListSource([b, c])
    settings = Settings(_env_file=None, retrieval={"sources": ["dense", "bm25"], "top_n": 7})
    pipe = SearchPipeline(settings, {"dense": s1, "bm25": s2})
    assert [x.chunk_id for x in pipe.search("q", 2)] == ["b", "a"]  # b 两路都召回，排第一
    assert s1.calls[0][0] == 7 and s2.calls[0][0] == 7  # 多路时每路取 top_n 个候选
    assert pipe.config["fusion"] == "rrf" and pipe.config["k_rrf"] == 60 and pipe.config["top_n"] == 7



@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("贵州茅台2024年研发投入合计是多少？", "研发投入合计是多少？"),
        ("茅台 2025 年的营业收入", "的营业收入"),  # 别名
        ("2024年贵州茅台和五粮液谁的研发投入更高？", "和谁的研发投入更高？"),  # 多家公司
        ("五粮液2024年度的利润分配预案是什么？", "的利润分配预案是什么？"),  # "年度"一并去掉
        ("贵州茅台2024年", "贵州茅台2024年"),  # 去完为空时返回原问题
        ("招商银行2025年的净息差是多少？", "招商银行的净息差是多少？"),  # 识别不到的公司保留
    ],
)
def test_strip_entities(query: str, expected: str) -> None:
    assert PARSER.strip_entities(query) == expected


def test_strip_requires_filter() -> None:
    with pytest.raises(ValueError, match="use_filter"):
        Settings(_env_file=None, retrieval={"strip_entities": True})


def test_pipeline_passes_stripped_text_to_sources() -> None:
    src = ListSource([])
    src.search = lambda q, k, flt=None: (src.calls.append((q, flt)), [])[1]
    settings = Settings(_env_file=None, retrieval={"use_filter": True, "strip_entities": True})
    pipe = SearchPipeline(settings, {"dense": src}, parser=PARSER)
    pipe.search("贵州茅台2025年研发投入合计是多少？", 5)
    text, flt = src.calls[0]
    assert text == "研发投入合计是多少？"  # 检索文本去掉了公司和年份
    assert flt == QueryFilter(("600519",), (2025,))  # 过滤条件仍来自原问题
    assert pipe.config["retriever"] == "filter+strip+dense"



# ---------------- 重排 ----------------


def test_rerank_by_scores_stable() -> None:
    from retrieval.rerank import rerank_by_scores

    a, b, c = (replace(_chunk(i, t), chunk_id=t) for i, t in enumerate("abc"))
    out = rerank_by_scores([a, b, c], [0.1, 0.9, 0.1])
    assert [x.chunk_id for x in out] == ["b", "a", "c"]  # a、c 同分，保持原顺序
    assert out[0].extra["rerank_score"] == 0.9


class FakeReranker:
    name = "fake-reranker"

    def __init__(self):
        self.seen = None

    def rerank(self, query, chunks):
        self.seen = (query, [c.chunk_id for c in chunks])
        return list(reversed(chunks))  # 倒序，便于验证确实用了重排结果


def test_pipeline_reranks_top_n_with_search_text() -> None:
    chunks = [replace(_chunk(i, t), chunk_id=t) for i, t in enumerate("abcdef")]
    src = ListSource(chunks)
    rr = FakeReranker()
    settings = Settings(_env_file=None, retrieval={"use_filter": True, "strip_entities": True, "top_n": 4, "rerank": True})
    pipe = SearchPipeline(settings, {"dense": src}, parser=PARSER, reranker=rr)
    out = pipe.search("贵州茅台2025年研发投入是多少？", 2)
    assert src.calls[0][0] == 4  # 需要重排时召回 top_n 个候选
    assert rr.seen == ("研发投入是多少？", ["a", "b", "c", "d"])  # 重排用去实体后的检索文本
    assert [c.chunk_id for c in out] == ["d", "c"]  # 取重排后的前 top_k
    assert pipe.config["reranker"] == "fake-reranker" and pipe.config["top_n"] == 4


def test_reranker_api_provider_not_implemented() -> None:
    from retrieval.rerank import build_reranker

    with pytest.raises(NotImplementedError):
        build_reranker(Settings(_env_file=None, reranker={"provider": "api"}).reranker)
