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
    # 本地模型目录；为空时由 FlagEmbedding 按 model_name 从 HuggingFace 下载。
    # 网络受限时先把模型下载到该目录（见 README），相对路径以项目根目录为基准
    model_path: Path | None = Path("data/models/bge-m3")
    api_base: str | None = None  # provider=openai_compat 时必填
    api_key: SecretStr | None = None
    device: Device = "cpu"
    batch_size: int = Field(default=16, gt=0)

    @field_validator("api_base", "api_key", "model_path", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> object:
        # .env 里写 `EMBEDDING__API_BASE=`（空值）时读到的是空字符串，这里统一视为"未配置"
        return None if v == "" else v

    @property
    def model_source(self) -> str:
        """传给模型加载函数的名称或路径：本地目录存在时优先使用，否则用 model_name。

        在这里（而不是字段校验器里）解析相对路径：pydantic 默认不校验字段默认值，
        写在校验器里的话，默认值不会被转换成绝对路径。
        """
        if self.model_path is not None:
            path = self.model_path if self.model_path.is_absolute() else PROJECT_ROOT / self.model_path
            if path.exists():
                return str(path)
        return self.model_name

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

    # local：本地通过 FlagEmbedding 推理（实验阶段使用，结果可复现）；
    # api：托管的重排接口（预留开关，部署到无 GPU 服务器时再接入，见 docs/decisions.md）
    provider: Literal["local", "api"] = "local"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_path: Path | None = Path("data/models/bge-reranker-v2-m3")  # 本地模型目录
    device: Device = "cpu"


class RetrievalSettings(BaseModel):
    """检索流水线（retrieval/search.py）。阶段 5 的消融实验通过这些开关逐个切换，一次只改一个。"""

    # 查询理解：从问题中识别公司、年份，作为 Qdrant payload 过滤条件
    use_filter: bool = False
    # 公司别名 -> 股票代码。manifest 中的公司简称和股票代码会自动识别，这里只放额外的常用简称
    company_aliases: dict[str, str] = {
        "茅台": "600519", "宁德": "300750", "中芯": "688981",
        "CATL": "300750", "BYD": "002594", "SMIC": "688981",
    }  # fmt: skip
    # 召回通道；多于一路时用 RRF 融合
    sources: list[Literal["dense", "bm25", "sparse"]] = ["dense"]
    fusion: Literal["rrf"] = "rrf"
    k_rrf: int = Field(default=60, gt=0)  # RRF 平滑常数，取原论文默认值
    top_n: int = Field(default=50, gt=0)  # 每路召回的候选数（融合或重排之前）
    bm25_k1: float = Field(default=1.5, gt=0)  # 词频饱和参数
    bm25_b: float = Field(default=0.75, ge=0, le=1)  # 文档长度归一化强度
    rerank: bool = False

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, v: list[str]) -> list[str]:
        if not v or len(set(v)) != len(v):
            raise ValueError("sources 不能为空，且不能重复")
        return v


class StoreSettings(BaseModel):
    """存储服务地址。"""

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "report_chunks"
    # DSN 里可能带数据库密码，所以同样用 SecretStr
    postgres_dsn: SecretStr = SecretStr("postgresql://copilot@localhost:5432/copilot")
    redis_url: str = "redis://localhost:6379/0"


class ParseSettings(BaseModel):
    """解析器版本（阶段 5 消融变量之一）。"""

    parser: Literal["v0", "v1"] = "v0"  # v0：PyMuPDF 文本块，不做清洗；v1：v0 + 去页眉页脚
    # v1 的页眉页脚判定：位于页面顶部/底部 header_band 比例区域内，且（数字替换为占位符后）
    # 在不少于 header_min_page_ratio 比例的页面上重复出现的文本块
    header_band: float = Field(default=0.1, gt=0, lt=0.5)
    header_min_page_ratio: float = Field(default=0.3, gt=0, le=1)


class ChunkSettings(BaseModel):
    """切分策略（阶段 5 消融实验的主要变量）。"""

    chunker: Literal["fixed"] = "fixed"  # fixed：固定字符长度切分
    chunk_size: int = Field(default=500, gt=0)  # 每个 chunk 的最大字符数
    overlap: int = Field(default=50, ge=0)  # 相邻 chunk 重叠的字符数

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkSettings:
        # overlap >= chunk_size 时步长 <= 0，切分会原地打转
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        return self

    @property
    def key(self) -> str:
        """切分器配置的短标识，如 fixed_c500_o50（不含解析器版本，完整标识见 Settings.chunk_key）。"""
        return f"{self.chunker}_c{self.chunk_size}_o{self.overlap}"


class IndexSettings(BaseModel):
    """向量化与建索引。"""

    max_length: int = Field(default=1024, gt=0)  # 向量化时的最大 token 数，超出部分截断
    upsert_batch: int = Field(default=256, gt=0)  # 每批写入 Qdrant 的点数
    # 同时写入 bge-m3 稀疏向量（供 sparse 召回通道使用）；与 dense 一次前向得到，几乎不增加建索引时间
    with_sparse: bool = True


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


class FrozenGoldset(BaseModel):
    """一个已冻结的评测集版本。"""

    path: Path  # 相对路径以项目根目录为基准
    sha256: str  # 冻结时的文件指纹；文件内容与之不符时 run_eval 拒绝运行


class EvalSettings(BaseModel):
    """检索评测。"""

    # 已冻结的评测集版本登记表（新增版本只能追加，已有条目不再修改），见 eval/goldset_schema.md
    goldsets: dict[str, FrozenGoldset] = {
        "v1": FrozenGoldset(
            path=Path("eval/goldset.jsonl"),
            sha256="cc1f43caf4e3e629e27d266703912ca1df44d5d6b3a7c2182c17fb9167208d2e",
        ),
        # v2 = v1 原题 60 道 + 口语化改写 60 道（variant=paraphrase）
        "v2": FrozenGoldset(
            path=Path("eval/goldset_v2.jsonl"),
            sha256="c9cb0ba8be789ed36ee9ba82bef6680db7cfbfd8603fa99334e60e1e6ce2f816",
        ),
    }
    goldset_version: str = "v1"  # run_eval 未指定 --goldset 时使用的版本
    # 计算指标的 K 值；检索深度取其中最大值。log.csv 固定记录 hit@5 / recall@5 / mrr@10，所以必须包含 5 和 10
    k_values: list[int] = [1, 3, 5, 10]

    @field_validator("k_values")
    @classmethod
    def _check_k(cls, v: list[int]) -> list[int]:
        if not {5, 10} <= set(v) or min(v) < 1:
            raise ValueError("k_values 必须是正整数，且包含 5 和 10")
        return sorted(set(v))

    @model_validator(mode="after")
    def _check_default_version(self) -> EvalSettings:
        if self.goldset_version not in self.goldsets:
            raise ValueError(f"goldset_version={self.goldset_version} 不在已登记的版本 {sorted(self.goldsets)} 中")
        return self

    def goldset_file(self, version: str) -> Path:
        """某个版本评测集的绝对路径。"""
        path = self.goldsets[version].path
        return path if path.is_absolute() else PROJECT_ROOT / path


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
    parse: ParseSettings = ParseSettings()
    chunk: ChunkSettings = ChunkSettings()
    index: IndexSettings = IndexSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    cninfo: CninfoSettings = CninfoSettings()
    precheck: PrecheckSettings = PrecheckSettings()
    eval: EvalSettings = EvalSettings()
    log: LogSettings = LogSettings()

    def chunk_key(self) -> str:
        """切分结果的完整标识 = 解析器版本 + 切分器配置，如 v0_fixed_c500_o50。

        解析器或切分参数任一不同，chunk 就不同，文件名、chunk_id、集合名都随之不同，互不覆盖。
        """
        return f"{self.parse.parser}_{self.chunk.key}"

    def collection_name(self) -> str:
        """Qdrant 集合名 = 基础名 + 切分标识 + 向量模型，不同消融配置的索引可以并存。

        例：report_chunks__v0_fixed_c500_o50__bge-m3
        """
        model = self.embedding.model_name.split("/")[-1].lower()
        return f"{self.store.qdrant_collection}__{self.chunk_key()}__{model}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的配置实例（首次调用时读取，之后复用缓存）。

    测试中如果修改了环境变量，需要先调用 `get_settings.cache_clear()` 再重新获取。
    """
    return Settings()
