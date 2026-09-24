# 架构说明

> 占位文档，随各阶段完成逐步补充。

## 模块划分

| 目录 | 职责 | 状态 |
|---|---|---|
| `config.py` | 统一配置 | 阶段 0 已完成 |
| `common/` | 公共工具（结构化日志、trace_id、阶段计时） | 阶段 0 已完成 |
| `ingest/` | PDF 预检、解析、表格、OCR、切分、建索引 | 待实现 |
| `retrieval/` | Dense / Sparse / BM25 检索、RRF 融合、重排 | 待实现 |
| `generation/` | Prompt、答案生成、引用校验 | 待实现 |
| `eval/` | 评测集、检索与生成评测 | 待实现 |
| `agent/` | LangGraph Agent 与工具 | 待实现 |
| `api/` | FastAPI 服务 | 待实现 |
| `ui/` | Streamlit 前端 | 待实现 |

## 数据流

（阶段 4 完成基线后补充）
