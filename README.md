# report-copilot：上市公司年报智能分析助手

针对上市公司年报这类复杂 PDF 的问答助手：

- 答案带原文页码引用，没有证据就拒答；
- 计算和跨公司对比交给 Agent 调用工具完成；
- 全过程用自建评测集量化，所有指标都能追溯到 `experiments/log.csv`。

> 仅基于公开的定期报告，不提供投资建议。

## 本地开发

```bash
# 1. 安装 uv（任选其一；brew 访问 ghcr.io 慢时用 pip）
brew install uv
pip install uv

# 2. 创建虚拟环境并安装依赖（Python 3.11）
uv venv --python 3.11
uv sync

# 3. 准备配置
cp .env.example .env   # 按需填写模型地址和密钥

# 4. 运行测试
uv run pytest
```

## 模型与基础设施

```bash
# 向量模型 bge-m3（约 2.3GB）：放到 data/models/bge-m3/（EMBEDDING__MODEL_PATH）。
# 只需要这 12 个文件：config.json、pytorch_model.bin、tokenizer.json、tokenizer_config.json、
# special_tokens_map.json、sentencepiece.bpe.model、colbert_linear.pt、sparse_linear.pt、modules.json、
# sentence_bert_config.json、config_sentence_transformers.json、1_Pooling/config.json（不需要 onnx/ 目录）。
# 来源：HuggingFace BAAI/bge-m3，或国内镜像 ModelScope BAAI/bge-m3（文件相同，可用 HuggingFace 公布的 sha256 校验）。
# 目录不存在时，FlagEmbedding 会按 EMBEDDING__MODEL_NAME 自动从 HuggingFace 下载。

# 启动 Qdrant、Postgres、Redis（开发用；密码写在 .env 的 POSTGRES_PASSWORD、REDIS_PASSWORD 中，模板见 .env.example）
docker compose --env-file .env -f deploy/docker-compose.dev.yml up -d
```

## 数据准备

```bash
# 从巨潮资讯网下载年报（写入 data/raw/，登记到 data/manifest.csv）
uv run python -m ingest.download --codes 600519,000858,000568,300750,002594,688981 --years 2024,2025

# PDF 预检（逐页明细 -> data/parsed/precheck/，汇总 -> data/precheck_summary.csv）
uv run python -m ingest.precheck

# 解析 -> 切分 -> 建索引（切分参数见 config.chunk；Apple 芯片可加 EMBEDDING__DEVICE=mps，快约 8 倍）
uv run python -m ingest.parse
uv run python -m ingest.chunk
EMBEDDING__DEVICE=mps uv run python -m ingest.index

# 财务指标表（阶段 7）：从"主要会计数据和财务指标"表抽取 9 个核心指标，逐条原文校验后写入 Postgres
uv run python -m agent.fin_db build
```

## Agent（阶段 7）

```bash
# 命令行试用：同一次运行内的多个问题属于同一对话（第二个问题是追问）
uv run python -m agent.graph "比亚迪2024年归属于上市公司股东的净利润占营业收入的比例是多少？" "那宁德时代呢？"
```

固定骨架 + 工具循环：先用 E013 检索取 10 个 chunk，再由模型决定是否调用工具补充证据或计算，最后作答并校验引用。
三个工具：`get_financial_metric`（财务指标表）、`search_reports`（检索）、`calculator`（ast 白名单 + Decimal 的安全计算器）。
多轮对话由 PostgresSaver 保存状态，追问先结合历史改写成完整问题。设计见 `docs/decisions.md` D-026。

## API 服务（阶段 8）

```bash
# 启动（检索配置与评测相同；首次启动加载 bge-m3 约十几秒）。接口文档：http://127.0.0.1:8000/docs
env PARSE__PARSER=v1 EMBEDDING__DEVICE=mps RETRIEVAL__USE_FILTER=true RETRIEVAL__STRIP_ENTITIES=true \
    RETRIEVAL__SOURCES='["dense","bm25","sparse"]' \
    uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log

# 流式问答（SSE）：start → rewrite → retrieve → decide → tool… → answer → done
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream -H 'Content-Type: application/json' \
    -d '{"question": "宁德时代2025年营业收入同比增长了多少？", "thread_id": "demo-1"}'
# 同一 thread_id 追问；查看对话历史
curl -X POST http://127.0.0.1:8000/v1/chat -H 'Content-Type: application/json' -d '{"question": "那比亚迪呢？", "thread_id": "demo-1"}'
curl http://127.0.0.1:8000/v1/threads/demo-1
```

鉴权（`API__KEYS`，请求头 `X-API-Key`）、限流（Redis，默认每客户端每分钟 30 次）、并发上限（默认 8，满了返回 503）都在 `.env` 中配置。
错误统一返回 `{"error": {code, message, trace_id}}`，响应头带 `X-Trace-Id`，可以用它在服务端日志中查到这次请求的全部记录。设计见 D-029。

## 评测

```bash
# 校验评测集（逐题回到 PDF 原文核对 span）
uv run python -m eval.validate_goldset

# 检索评测。合成检索器用 --no-log，正式实验用 --desc 写明改动，结果追加到 experiments/log.csv
uv run python -m eval.run_eval --retriever oracle --no-log
uv run python -m eval.run_eval --retriever <name> --desc "改动说明"
```

## 结果

### 留出集 v3（最终报告数字）

v3 共 80 题，按固定种子抽样页出题，不参与任何调参。阶段 6 结束时按预先登记各运行一次（D-024、D-025）；
阶段 7 的 Agent 在开发集上确定保留后，按预先登记再运行一次（v3 的第二次使用，D-027、D-028）。

| | 检索 Recall@5 | 检索 Hit@5 | 生成评审正确率 | 数值正确率 | 引用可核验率 | 拒答召回率 | 误拒率 |
|---|---|---|---|---|---|---|---|
| v0 朴素基线（E020 / G006） | 0.4142 | 0.4706 | 0.5687 | 0.3673 | 0.9733 | 1.0000 | 0.4118 |
| 阶段 6 系统（E021 / G005） | 0.8554 | 0.9265 | 0.8625 | 0.7143 | 1.0000 | 1.0000 | 0.0735 |
| **最终系统：Agent（E021 / G008）** | **0.8554** | **0.9265** | **0.9375** | **0.7959** | 0.9928 | **1.0000** | **0.0000** |

- 检索和生成相对基线的提升都显著（95% CI 不跨 0）：Recall@5 +0.44；评审正确率 G005 相对 G006 +0.29。
- Agent 相对阶段 6 系统：评审正确率 +0.075（95% CI [+0.013, +0.144]，不跨 0；但符号检验 p=0.15，9 题变好、3 题变差，证据强度一般）；
  误拒率从 0.0735 降到 0。
- 主要失败：表格读错行/列（BC-012）；Agent 引入的新问题——工具只有"营业收入"，模型把问题里的"营业总收入"换成了它（BC-013）。
- **局限**：v3 全部由 AI 出题，开发者未抽查；80 题的置信区间约 ±0.065；v3 已使用两次，第二次的结论不再用于调参。

### 多轮对话（阶段 7）

15 组 AI 构造的对话，最后一轮是"那五粮液呢？"这类省略式追问，只评最后一轮（D-027）：
带记忆（D001，PostgresSaver）评审正确率 1.0000，不带记忆（D002）0.1333，差值 +0.87（95% CI [+0.67, +1.00]）。
不带记忆时模型几乎全部拒答，没有乱猜。对话集规模小、由 AI 构造，只说明机制可用。

### 开发集（v1/v2，用于做所有取舍）

检索消融（阶段 5）与生成实验（阶段 6）都在开发集上进行，结论见 D-019、D-023。最终配置在开发集 v2 上：
检索 Recall@5 0.7869（E013）、生成评审正确率 0.9042（G002）；阶段 7 的 Agent（G007）0.9667，相对 G002 确定提升
（+0.0625，95% CI [+0.025, +0.104]，D-028）。生成侧开发集数字高于留出集，说明存在"在开发集上挑选"的乐观偏差。

```bash
# 检索评测 / 生成评测 / 两次实验的显著性对比
uv run python -m eval.run_eval --retriever pipeline --goldset v2 --desc "改动说明"
uv run python -m eval.run_gen_eval --goldset v2 --desc "改动说明"
uv run python -m eval.run_gen_eval --goldset v2 --answerer agent --desc "改动说明"   # Agent
uv run python -m eval.run_dialog_eval --mode memory --desc "改动说明"               # 多轮对话
uv run python -m eval.compare_runs G002 G003
```

## 目录结构

见 [AGENTS.md](AGENTS.md) 第 3 节；技术选择的理由见 [docs/decisions.md](docs/decisions.md)。

## 进度

见 [AGENTS.md](AGENTS.md) 第 7 节。
