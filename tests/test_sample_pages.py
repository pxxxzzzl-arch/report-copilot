"""eval/sample_pages.py 的单元测试。"""

import json
from pathlib import Path

from eval.sample_pages import eligible_pages, gold_pages, sample


def test_gold_pages_includes_alt(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    item = {"gold": [{"doc_id": "A", "page": 2, "span": "x", "alt": [{"doc_id": "B", "page": 7}]}]}
    path.write_text(json.dumps(item) + "\n" + json.dumps({"gold": []}) + "\n", encoding="utf-8")
    assert gold_pages([path]) == {("A", 2), ("B", 7)}


def test_eligible_pages_excludes_gold_and_short_pages() -> None:
    chars = {1: 50, 2: 800, 3: 900, 4: 199, 5: 200}
    assert eligible_pages(5, chars, "A", {("A", 3), ("B", 2)}, min_chars=200) == [2, 5]


def test_sample_reproducible_and_independent_per_doc() -> None:
    cand = list(range(1, 101))
    a1 = sample(cand, 10, seed=20260928, doc_id="A")
    assert a1 == sample(cand, 10, seed=20260928, doc_id="A")  # 同种子同文档：结果相同
    assert a1 != sample(cand, 10, seed=20260928, doc_id="B")  # 不同文档：各自独立
    assert a1 != sample(cand, 10, seed=1, doc_id="A")  # 不同种子：结果不同
    assert len(set(a1)) == 10 and set(a1) <= set(cand)
    assert sample([1, 2], 10, seed=1, doc_id="A") in ([1, 2], [2, 1])  # 候选不足时全部返回
    assert cand == list(range(1, 101))  # 不修改传入的列表
