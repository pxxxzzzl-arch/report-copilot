"""统一配置入口。

所有模型名、服务地址、密钥、路径都从这里读取，业务代码中不写死任何配置值。

读取优先级（高 -> 低）：
    1. 进程环境变量，例如 `export LLM__MODEL=xxx`
    2. 项目根目录下的 `.env` 文件
    3. 本文件中字段的默认值

嵌套字段用双下划线 `__` 分隔：环境变量 `LLM__BASE_URL` 对应 `settings.llm.base_url`。

用法：
    from config import get_settings
    settings = get_settings()
    settings.llm.model
    settings.llm.api_key.get_secret_value()   # 只在真正发请求时取明文
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 = 本文件所在目录。
# 相对路径都以它为基准解析，这样无论从哪个工作目录启动脚本，路径都一致。
PROJECT_ROOT = Path(__file__).resolve().parent

# 推理设备：Apple 芯片用 mps，NVIDIA 显卡用 cuda
Device = Literal["cpu", "mps", "cuda"]


class PathSettings(BaseModel):
    """数据与实验相关的路径。"""

    # pydantic 默认不校验"默认值"，不开这个选项的话，下面的默认相对路径不会经过 _resolve_relative
    model_config = ConfigDict(validate_default=True)

    raw_dir: Path = Path("data/raw")  # 原始年报 PDF
    parsed_dir: Path = Path("data/parsed")  # 解析中间结果
    manifest_path: Path = Path("data/manifest.csv")  # 文档清单
    precheck_summary_path: Path = Path("data/precheck_summary.csv")  # PDF 预检汇总（每文档一行）
    experiments_dir: Path = Path("experiments")  # 实验记录 log.csv / badcases.md / runs/
    goldset_path: Path = Path("eval/goldset.jsonl")  # 评测集

    @field_validator("*")
    @classmethod
    def _resolve_relative(cls, v: Path) -> Path:
        # 相对路径统一转成基于项目根目录的绝对路径；绝对路径保持不变
        return v if v.is_absolute() else PROJECT_ROOT / v


class LLMSettings(BaseModel):
    """大模型：任意 OpenAI 兼容接口（开发期用云 API，压测期换成自部署 vLLM）。"""

    base_url: str = "http://localhost:8000/v1"  # vLLM 默认的 OpenAI 兼容地址
    # 密钥用 SecretStr：打印对象或写日志时显示为 '**********'，防止泄露
    api_key: SecretStr = SecretStr("")
    model: str = ""
    timeout_s: float = Field(default=60.0, gt=0)


class EmbeddingSettings(BaseModel):
    """向量模型：默认本地 bge-m3，可切换为 OpenAI 兼容的 Embedding API。"""

    # 字段名 model_name 以 "model_" 开头，与 pydantic 保留的命名空间冲突会告警，这里关闭该检查
    model_config = ConfigDict(protected_namespaces=())

    provider: Literal["local", "openai_compat"] = "local"
    model_name: str = "BAAI/bge-m3"
    api_base: str | None = None  # provider=openai_compat 时必填
    api_key: SecretStr | None = None
    device: Device = "cpu"
    batch_size: int = Field(default=16, gt=0)

    @field_validator("api_base", "api_key", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        # .env 里写 `EMBEDDING__API_BASE=`（空值）时读到的是空字符串，这里统一视为"未配置"
        return None if v == "" else v

    @model_validator(mode="after")
    def _check_api_base(self) -> EmbeddingSettings:
        # 在启动时就发现配置错误，而不是等到第一次调用 API 才报错
        if self.provider == "openai_compat" and not self.api_base:
            raise ValueError("EMBEDDING__PROVIDER=openai_compat 时必须配置 EMBEDDING__API_BASE")
        return self


class RerankerSettings(BaseModel):
    """重排模型。"""

    # 字段名 model_name 以 "model_" 开头，与 pydantic 保留的命名空间冲突会告警，这里关闭该检查
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: Device = "cpu"


class StoreSettings(BaseModel):
    """存储服务地址。"""

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "report_chunks"
    # DSN 里可能带数据库密码，所以同样用 SecretStr
    postgres_dsn: SecretStr = SecretStr("postgresql://copilot@localhost:5432/copilot")
    redis_url: str = "redis://localhost:6379/0"


class CninfoSettings(BaseModel):
    """巨潮资讯网（年报下载来源）。"""

    # 股票代码 -> orgId 的映射表（文件名是 szse，但实际包含沪深两市全部 A 股）；
    # 查询公告接口必须同时传代码和 orgId
    stock_list_url: str = "http://www.cninfo.com.cn/new/data/szse_stock.json"
    query_url: str = "http://www.cninfo.com.cn/new/hisAnnouncement/query"  # 公告查询接口
    static_base_url: str = "http://static.cninfo.com.cn/"  # PDF 地址 = 该前缀 + adjunctUrl
    request_interval_s: float = Field(default=1.0, ge=0)  # 两次请求的最小间隔，避免给对方服务器造成压力
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=1)


class PrecheckSettings(BaseModel):
    """PDF 预检阈值。初始值为经验起点，需结合真实数据的分布再校准。"""

    min_text_chars: int = Field(default=30, ge=0)  # 页内有效字符数低于它，视为"几乎没有文字"
    scan_image_ratio: float = Field(default=0.5, ge=0, le=1)  # 图片覆盖面积占比超过它，且文字很少 -> 扫描页
    garbled_ratio: float = Field(default=0.1, ge=0, le=1)  # 乱码字符占比超过它 -> 乱码页


class EvalSettings(BaseModel):
    """检索评测。"""

    # 冻结的评测集版本及其 sha256：文件内容与之不符时 run_eval 拒绝运行，保证 log.csv 中同版本的结果可比
    goldset_version: str = "v1"
    goldset_sha256: str = "cc1f43caf4e3e629e27d266703912ca1df44d5d6b3a7c2182c17fb9167208d2e"
    # 计算指标的 K 值；检索深度取其中最大值。log.csv 固定记录 hit@5 / recall@5 / mrr@10，所以必须包含 5 和 10
    k_values: list[int] = [1, 3, 5, 10]

    @field_validator("k_values")
    @classmethod
    def _check_k(cls, v: list[int]) -> list[int]:
        if not {5, 10} <= set(v) or min(v) < 1:
            raise ValueError("k_values 必须是正整数，且包含 5 和 10")
        return sorted(set(v))


class LogSettings(BaseModel):
    """日志。"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = True  # True: 单行 JSON；False: 人类可读文本

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, v: object) -> object:
        # 允许 .env 里写小写的 info / debug
        return v.upper() if isinstance(v, str) else v


class Settings(BaseSettings):
    """全局配置对象，按模块分组。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",  # 用绝对路径，避免依赖启动时的工作目录
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # LLM__MODEL -> settings.llm.model
        extra="ignore",  # .env 中出现未定义的变量时忽略，不报错
    )

    paths: PathSettings = PathSettings()
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    reranker: RerankerSettings = RerankerSettings()
    store: StoreSettings = StoreSettings()
    cninfo: CninfoSettings = CninfoSettings()
    precheck: PrecheckSettings = PrecheckSettings()
    eval: EvalSettings = EvalSettings()
    log: LogSettings = LogSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的配置实例（首次调用时读取，之后复用缓存）。

    测试中如果修改了环境变量，需要先调用 `get_settings.cache_clear()` 再重新获取。
    """
    return Settings()
