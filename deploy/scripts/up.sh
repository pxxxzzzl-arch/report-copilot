#!/usr/bin/env bash
# 本地一键运行（阶段 10，D-031）：检查前提 → 构建镜像 → 启动基础设施与 api/ui → 等所有服务就绪。
#
# 用法（任意目录执行均可）：
#   deploy/scripts/up.sh          # 启动；完成后浏览器打开 http://localhost:8501
#   deploy/scripts/up.sh down     # 停止并删除容器（数据保留在 deploy/volumes/）
#
# 前提：已按 README 准备好数据（下载 → 解析 → 切分 → 建索引 → 建财务指标表）。年报 PDF 和模型权重不在 Git 中。
set -euo pipefail

cd "$(dirname "$0")/../.."   # 项目根目录
COMPOSE=(docker compose --env-file .env -f deploy/docker-compose.dev.yml -f deploy/docker-compose.yml)

if [ "${1:-}" = "down" ]; then
  "${COMPOSE[@]}" down
  exit 0
fi

fail() { echo "✗ $1" >&2; exit 1; }

docker info >/dev/null 2>&1 || fail "Docker 未运行，请先启动 Docker Desktop"
[ -f .env ] || fail "缺少 .env：cp .env.example .env，并填写大模型地址、密钥和数据库口令"
# 与 api 的部署配置对应：parser v1 + fixed 500/50 的切分结果、bge-m3 模型（README「数据准备」「模型与基础设施」）
for p in data/manifest.csv data/raw data/parsed/chunks/v1_fixed_c500_o50.jsonl data/models/bge-m3/config.json; do
  [ -e "$p" ] || fail "缺少 $p，请先按 README「数据准备」准备数据和模型（解析、切分、建索引都要加 PARSE__PARSER=v1）"
done

echo "→ 构建镜像并启动服务（首次加载向量模型约需 1 分钟）"
"${COMPOSE[@]}" up -d --build --wait

# /readyz 已确认向量集合存在（否则 --wait 会失败），但不检查财务指标表，这里补上（get_financial_metric 工具依赖它）
n=$("${COMPOSE[@]}" exec -T postgres psql -U copilot -d copilot -tAc "select count(*) from fin_metrics" 2>/dev/null || echo 0)
[ "${n// /}" -gt 0 ] 2>/dev/null || echo "⚠ 财务指标表为空，请先执行：uv run python -m agent.fin_db build"

echo "✓ 已就绪：http://localhost:8501（API 文档：http://localhost:8000/docs）"
echo "  停止：deploy/scripts/up.sh down"
