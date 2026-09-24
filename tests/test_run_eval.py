"""eval/run_eval.py 与 eval/sanity_retrievers.py 的单元测试（不读取真实 PDF）。"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval.run_eval import (
    append_log,
    build_log_row,
    evaluate,
    file_sha256,
    load_frozen_goldset,
    next_exp_id,
    percentile,
    write_details,
)
from eval.sanity_retrievers import OracleRetriever, RandomPageRetriever
from ingest.download import ManifestRow

HEADER = (
    "exp_id,date,git_commit,改动说明,parser,chunker,chunk_size,overlap,retriever,fusion,k_rrf,reranker,"
    "top_k,top_n,goldset_version,n_questions,hit@5,recall@5,mrr@10,备注"
)
KS = [1, 3, 5, 10]

ITEMS = [
    {"id": "q001", "type": "table_number", "question": "Q1", "answer": "", "gold": [{"doc_id": "A", "page": 2, "span": "1,234"}]},
    {"id": "q002", "type": "compare", "question": "Q2", "answer": "",
     "gold": [{"doc_id": "A", "page": 1, "span": "aaaa"}, {"doc_id": "B", "page": 1, "span": "bbbb"}]},
    {"id": "q003", "type": "out_of_scope", "question": "Q3", "answer": "", "gold": []},
]  # fmt: skip
PAGES = {"A": ["aaaa", "收入 1, 234 元"], "B": ["bbbb", "zzzz"]}


def page_texts(doc_id: str) -> list[str]:
    return PAGES[doc_id]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str


class FakeRetriever:
    """Q1：第 2 名命中；Q2：第 1 名命中 A，B 没找到。"""

    name = "fake"
    config = {"chunker": "fixed", "chunk_size": 512, "top_n": 5}

    def search(self, query: str, top_k: int) -> list[Chunk]:
        if query == "Q1":
            return [Chunk("x", "A", 1, 1, "aaaa"), Chunk("y", "A", 2, 2, "1,234")]
        return [Chunk("z", "A", 1, 1, "aaaa")]


# ---------------- 评测 ----------------


def test_evaluate_scores_and_skips_out_of_scope() -> None:
    records, summary = evaluate(FakeRetriever(), ITEMS, KS)
    assert [r["id"] for r in records] == ["q001", "q002"]
    assert summary["n_questions"] == 2 and summary["n_out_of_scope_skipped"] == 1

    q1, q2 = (r["scores"] for r in records)
    assert (q1["hit@1"], q1["hit@5"], q1["mrr@10"], q1["recall@5"]) == (0, 1, 0.5, 1.0)
    assert (q2["hit@1"], q2["recall@5"], q2["mrr@10"]) == (1, 0.5, 1.0)
    # 总体 = 两题平均：hit@1 = (0+1)/2，recall@5 = (1+0.5)/2
    assert summary["overall"]["hit@1"] == 0.5
    assert summary["overall"]["recall@5"] == 0.75
    assert summary["by_type"]["compare"]["n"] == 1
    # 明细里记录了每个结果命中的 gold 下标
    assert [x["hits"] for x in records[0]["retrieved"]] == [[], [0]]


def test_oracle_retriever_scores_perfectly() -> None:
    _, summary = evaluate(OracleRetriever(ITEMS, page_texts), ITEMS, KS)
    for m in ("hit@1", "recall@10", "mrr@10", "ndcg@10"):
        assert summary["overall"][m] == 1.0, m


def test_oracle_dedups_same_page() -> None:
    items = [{"question": "Q", "gold": [{"doc_id": "A", "page": 1, "span": "a"}, {"doc_id": "A", "page": 1, "span": "b"}]}]
    assert len(OracleRetriever(items, page_texts).search("Q", 10)) == 1


def test_random_retriever_is_deterministic() -> None:
    manifest = {d: ManifestRow(d, "", "", 0, "", 2, "", "", "", "") for d in PAGES}
    r = RandomPageRetriever(manifest, page_texts, seed=7)
    first = [c.chunk_id for c in r.search("Q1", 3)]
    assert first == [c.chunk_id for c in r.search("Q1", 3)]  # 同一问题结果相同
    assert len(set(first)) == 3  # 不重复抽样


@pytest.mark.parametrize(("p", "expected"), [(50, 3), (95, 5), (100, 5), (1, 1)])
def test_percentile_nearest_rank(p: float, expected: float) -> None:
    # 排序后 [1,2,3,4,5]：p50 -> 第 ceil(2.5)=3 个 = 3；p95 -> 第 ceil(4.75)=5 个 = 5
    assert percentile([5, 1, 4, 2, 3], p) == expected


def test_percentile_empty() -> None:
    assert percentile([], 50) is None


# ---------------- 评测集冻结 ----------------


def test_load_frozen_goldset_checks_sha(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in ITEMS), encoding="utf-8")
    assert len(load_frozen_goldset(path, file_sha256(path))) == 3
    with pytest.raises(ValueError, match="sha256"):
        load_frozen_goldset(path, "0" * 64)


# ---------------- 实验记录 ----------------


def _log(tmp_path: Path) -> Path:
    path = tmp_path / "log.csv"
    path.write_text(HEADER + "\n", encoding="utf-8")
    return path


def test_next_exp_id(tmp_path: Path) -> None:
    path = _log(tmp_path)
    assert next_exp_id(tmp_path / "missing.csv") == "E001"
    assert next_exp_id(path) == "E001"
    with path.open("a", encoding="utf-8") as f:
        f.write("E001" + "," * 19 + "\n" + "E007" + "," * 19 + "\n")
    assert next_exp_id(path) == "E008"  # 取最大编号 + 1


def test_build_and_append_log_row(tmp_path: Path) -> None:
    path = _log(tmp_path)
    _, summary = evaluate(FakeRetriever(), ITEMS, KS)
    header = HEADER.split(",")
    row = build_log_row(header, "E001", FakeRetriever(), summary, KS, "基线", "", "v1", "abc123-dirty")
    append_log(path, row)

    with path.open(encoding="utf-8", newline="") as f:
        (back,) = list(csv.DictReader(f))
    assert back["exp_id"] == "E001" and back["git_commit"] == "abc123-dirty"
    assert back["retriever"] == "fake"  # config 未提供时用检索器名字
    assert back["chunker"] == "fixed" and back["chunk_size"] == "512" and back["top_n"] == "5"
    assert back["top_k"] == "10"  # config 未提供时用检索深度 max(k)
    assert back["fusion"] == ""  # 未提供的配置列留空
    assert back["n_questions"] == "2"
    assert back["hit@5"] == "1.0000" and back["recall@5"] == "0.7500"
    assert back["改动说明"] == "基线" and "E001.jsonl" in back["备注"]


def test_write_details_roundtrip(tmp_path: Path) -> None:
    records, summary = evaluate(FakeRetriever(), ITEMS, KS)
    path = tmp_path / "runs" / "E001.jsonl"
    write_details(path, {"exp_id": "E001", "summary": summary}, records)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["meta"]["exp_id"] == "E001"
    assert [r["id"] for r in lines[1:]] == ["q001", "q002"]
