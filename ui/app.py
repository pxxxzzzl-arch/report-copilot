"""Streamlit 前端（阶段 9）：问答 + 流式进度 + 点击引用查看年报原页（高亮摘录位置）。

启动（先启动 API 服务，见 README）：
    uv run streamlit run ui/app.py
    浏览器打开 http://localhost:8501

布局：侧边栏（新建对话、服务状态、语料清单）｜左：对话｜右：原文查看器。
Streamlit 的运行方式：每次交互（提问、点按钮）都会把本脚本从头执行一遍，跨次执行的数据放在 st.session_state 里：
    messages  对话消息列表 [{"role", "content", "response", "progress", "error"}]
    thread_id API 返回的对话编号（第一次提问后才有；新建对话时清空）
    viewer    右栏正在查看的页面 {"doc_id", "page", "quote", "title"}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# streamlit run 只把脚本所在目录（ui/）加入 sys.path；把项目根目录也加进去，才能 import config、ui.client
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_settings  # noqa: E402  （必须在上面调整 sys.path 之后导入）
from ui.client import ApiClient, ApiError, describe_event  # noqa: E402

st.set_page_config(page_title="年报智能分析助手", page_icon="📊", layout="wide")


@st.cache_resource
def default_client() -> ApiClient:
    cfg = get_settings().ui
    return ApiClient(cfg.api_base, cfg.api_key.get_secret_value(), cfg.timeout_s)


def client() -> ApiClient:
    # 测试（streamlit.testing.v1.AppTest）通过 session_state["client"] 注入假客户端
    return st.session_state.get("client") or default_client()


@st.cache_data(max_entries=64, show_spinner=False)
def load_page(_api: ApiClient, doc_id: str, page: int, quote: str):
    """页面截图按 (文档, 页码, 摘录) 缓存；参数名以下划线开头的 _api 不参与缓存键。"""
    return _api.page_image(doc_id, page, quote)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("viewer", None)


def new_chat() -> None:
    st.session_state.messages, st.session_state.thread_id, st.session_state.viewer = [], None, None


def open_viewer(doc_id: str, page: int, quote: str, title: str) -> None:
    st.session_state.viewer = {"doc_id": doc_id, "page": page, "quote": quote, "title": title}


# ---------------- 侧边栏 ----------------


def sidebar() -> None:
    with st.sidebar:
        st.title("📊 年报智能分析助手")
        st.caption("答案只依据年报原文，附页码引用；没有证据就拒答。不提供投资建议。")
        st.button("＋ 新建对话", on_click=new_chat, width="stretch")
        if st.session_state.thread_id:
            st.caption(f"对话编号：`{st.session_state.thread_id}`")

        ready = client().ready()
        status = ready.get("status")
        label = {"ok": "🟢 服务正常", "degraded": "🟠 部分组件不可用", "unreachable": "🔴 无法连接 API"}.get(status, status)
        with st.expander(label, expanded=status != "ok"):
            for name, s in ready.get("components", {}).items():
                st.write(f"{'✅' if s == 'ok' else '❌'} {name}")
            if status == "unreachable":
                st.caption(f"请确认 API 已启动：{get_settings().ui.api_base}")

        if status != "unreachable":
            try:
                docs = client().documents()
            except ApiError:
                docs = []
            if docs:
                with st.expander(f"语料：{len({d['company'] for d in docs})} 家公司 · {len(docs)} 份年报"):
                    for d in docs:
                        st.write(f"{d['company']}（{d['stock_code']}）{d['year']} 年报 · {d['n_pages']} 页")


# ---------------- 对话 ----------------


def render_answer(msg: dict, idx: int) -> None:
    """渲染一条助手消息：答案、引用（可点击查看原文）、取证过程、全部证据。"""
    if msg.get("error"):
        st.error(msg["error"])
        return
    resp = msg["response"]
    if not resp["format_ok"]:
        st.warning("模型输出格式异常，未能生成答案，请重试或换个问法。")
    elif resp["refused"]:
        st.info(resp["answer"] or "无法回答")
    else:
        st.markdown(resp["answer"])

    evidence = {e["id"]: e for e in resp["evidence"]}
    cites = resp["citations"]
    if cites and not resp["refused"]:
        st.markdown("**引用**")
        for j, c in enumerate(cites):
            ev = evidence.get(c["ref"])
            quote = c["quote"].replace("\n", " ")
            short = quote if len(quote) <= 60 else quote[:60] + "…"
            if not (c["verified"] and ev):
                # 未通过核验的引用：摘录不在所引证据中，不能当作依据，不提供查看原文
                st.caption(f":gray[[{c['ref']}] 「{short}」 ⚠️ 未通过核验]")
                continue
            page = c["page"] or c["page_start"]
            title = f"{ev['company']} {ev['year']} 年报 · 第 {page} 页"
            left, right = st.columns([3, 1], vertical_alignment="center")
            left.markdown(f"**[{c['ref']}]** {title}  \n「{short}」")
            right.button("查看原文", key=f"cite-{idx}-{j}", on_click=open_viewer,
                         args=(c["doc_id"], page, c["quote"], f"[{c['ref']}] {title}"))  # fmt: skip

    with st.expander(f"取证过程（{resp['steps']} 步，{resp['latency_ms'] / 1000:.1f} 秒）"):
        for line in msg.get("progress", []):
            st.write(line)
    with st.expander(f"全部证据（{len(resp['evidence'])} 条）"):
        for e in resp["evidence"]:
            pages = f"{e['page_start']}" if e["page_start"] == e["page_end"] else f"{e['page_start']}-{e['page_end']}"
            st.markdown(f"**[{e['id']}]** {e['company']} {e['year']} 年报 第 {pages} 页 · `{e['source']}`")
            st.caption(e["text"][:300] + ("…" if len(e["text"]) > 300 else ""))


def ask(question: str) -> None:
    """提交问题：流式显示进度，结束后把完整消息存进 session_state。"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    msg: dict = {"role": "assistant", "content": "", "response": None, "progress": [], "error": None}
    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        with st.status("正在分析…", expanded=True) as status:
            try:
                for event, data in client().stream_chat(question, st.session_state.thread_id):
                    if event == "start":
                        st.session_state.thread_id = data["thread_id"]
                    elif event == "answer":
                        msg["response"] = data
                    elif event == "error":
                        err = data["error"]
                        msg["error"] = f"{err['message']}（trace_id={err.get('trace_id')}）"
                    line = describe_event(event, data, question)
                    if line:
                        msg["progress"].append(line)
                        status.write(line)
            except ApiError as e:
                msg["error"] = str(e)
            if msg["response"] is None and not msg["error"]:
                msg["error"] = "没有收到答案，连接可能已中断，请重试。"
            ok = msg["error"] is None
            status.update(label=f"{'完成' if ok else '出错'}（{time.perf_counter() - t0:.1f} 秒）",
                          state="complete" if ok else "error", expanded=not ok)  # fmt: skip
    st.session_state.messages.append(msg)
    st.rerun()  # 重新执行脚本，用 render_answer 统一渲染（带"查看原文"按钮）


# ---------------- 原文查看器 ----------------


def viewer() -> None:
    v = st.session_state.viewer
    if not v:
        st.caption("点击答案下方引用的「查看原文」，这里会显示年报原页，并用黄色框标出摘录的位置。")
        return
    st.markdown(f"**{v['title']}**")
    prev_col, next_col, close_col = st.columns(3)
    if prev_col.button("◀ 上一页", width="stretch", disabled=v["page"] <= 1):
        v["page"] -= 1
        st.rerun()
    if next_col.button("下一页 ▶", width="stretch"):
        v["page"] += 1
        st.rerun()
    if close_col.button("关闭", width="stretch"):
        st.session_state.viewer = None
        st.rerun()
    try:
        img = load_page(client(), v["doc_id"], v["page"], v["quote"])
    except ApiError as e:
        st.error(str(e))
        return
    if img.method == "none":
        st.warning(f"第 {v['page']} 页上没有定位到摘录（摘录可能跨页），可以翻页查看。")
    st.image(img.png, caption=f"{v['doc_id']} 第 {v['page']} 页", width="stretch")


# ---------------- 主流程 ----------------


def main() -> None:
    init_state()
    sidebar()
    chat_col, view_col = st.columns([3, 2], gap="large")
    with chat_col:
        # 先放一个容器承载对话，再放输入框：Streamlit 按调用顺序排版，流式回答写进这个容器，才会出现在输入框上方
        history = st.container()
        question = st.chat_input("输入问题，例如：宁德时代2025年研发投入是多少？", max_chars=500)
        with history:
            if not st.session_state.messages and not question:
                st.markdown("#### 可以这样问")
                st.caption("贵州茅台2025年营业收入是多少？ · 宁德时代2025年营业收入同比增长了多少？（再追问：那比亚迪呢？）"
                           " · 2024年五粮液和泸州老窖谁的经营活动现金流更多？")  # fmt: skip
            for i, m in enumerate(st.session_state.messages):
                with st.chat_message(m["role"]):
                    if m["role"] == "user":
                        st.markdown(m["content"])
                    else:
                        render_answer(m, i)
            if question:
                ask(question)
    with view_col:
        viewer()


main()
