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

# 启动 Qdrant（开发用）
docker compose -f deploy/docker-compose.dev.yml up -d
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
```

## 评测

```bash
# 校验评测集（逐题回到 PDF 原文核对 span）
uv run python -m eval.validate_goldset

# 检索评测。合成检索器用 --no-log，正式实验用 --desc 写明改动，结果追加到 experiments/log.csv
uv run python -m eval.run_eval --retriever oracle --no-log
uv run python -m eval.run_eval --retriever <name> --desc "改动说明"
```

## 目录结构

见 [AGENTS.md](AGENTS.md) 第 3 节；技术选择的理由见 [docs/decisions.md](docs/decisions.md)。

## 进度

见 [AGENTS.md](AGENTS.md) 第 7 节。
