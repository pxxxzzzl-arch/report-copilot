"""eval/run_dialog_eval.py 与对话集 dialogs_v1 的测试。"""


from config import PROJECT_ROOT, get_settings
from eval.run_dialog_eval import build_items, load_dialogs
from eval.run_eval import load_frozen_goldset


def test_build_items_takes_reference_from_target() -> None:
    dialogs = [{"id": "d01", "turns": ["A公司2024年X是多少？", "那B公司呢？"], "target": "q002"}]
    targets = {"q002": {"id": "q002", "type": "table_number", "question": "B公司2024年X是多少？", "answer": "1元",
                        "gold": [{"doc_id": "b", "page": 1, "span": "1"}]}}  # fmt: skip
    (it,) = build_items(dialogs, targets)
    # 题号、问题来自对话（最后一轮原话）；类型、参考答案、gold 来自目标题
    assert it["id"] == "d01" and it["question"] == "那B公司呢？"
    assert it["answer"] == "1元" and it["gold"] == targets["q002"]["gold"] and it["type"] == "table_number"


def test_dialogs_v1_frozen_and_well_formed() -> None:
    s = get_settings()
    frozen = s.eval.dialogs["dialogs_v1"]
    dialogs = load_dialogs(PROJECT_ROOT / frozen.path, frozen.sha256)  # 指纹不符会抛异常
    v1 = {it["id"]: it for it in load_frozen_goldset(s.eval.goldset_file("v1"), s.eval.goldsets["v1"].sha256)}
    assert len(dialogs) == 15 and len({d["id"] for d in dialogs}) == 15
    for d in dialogs:
        assert d["target"] in v1 and 2 <= len(d["turns"]) <= 3
        # 最后一轮必须是省略式追问：不能与目标题原文相同，否则测不出记忆的作用
        assert d["turns"][-1] != v1[d["target"]]["question"]
    assert sum(v1[d["target"]]["type"] == "out_of_scope" for d in dialogs) == 2
