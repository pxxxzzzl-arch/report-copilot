"""config.py 的单元测试。

所有测试都用 Settings(_env_file=None) 构造，不读取本机真实的 .env，保证结果可复现。
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import PROJECT_ROOT, Settings, get_settings

# config.py 中各配置分组对应的环境变量前缀
_PREFIXES = {"PATHS", "LLM", "EMBEDDING", "RERANKER", "STORE", "PARSE", "CHUNK", "INDEX", "RETRIEVAL", "CNINFO", "PRECHECK", "EVAL", "LOG"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉开发机 shell 里可能存在的同名环境变量，避免干扰测试。"""
    for key in list(os.environ):
        if key.split("__")[0].upper() in _PREFIXES:
            monkeypatch.delenv(key)


def test_defaults_load_without_env_file() -> None:
    s = Settings(_env_file=None)
    assert s.embedding.provider == "local"
    assert s.embedding.model_name == "BAAI/bge-m3"
    assert s.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert s.log.level == "INFO"


def test_relative_paths_resolved_against_project_root() -> None:
    s = Settings(_env_file=None)
    assert s.paths.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert s.paths.raw_dir.is_absolute()


def test_absolute_path_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATHS__RAW_DIR", str(tmp_path))
    assert Settings(_env_file=None).paths.raw_dir == tmp_path


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("EMBEDDING__BATCH_SIZE", "32")  # 字符串会被自动转换成 int
    monkeypatch.setenv("LOG__LEVEL", "debug")  # 小写也能识别
    s = Settings(_env_file=None)
    assert s.llm.model == "qwen2.5-7b-instruct"
    assert s.embedding.batch_size == 32
    assert s.log.level == "DEBUG"


def test_env_file_is_read(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM__MODEL=from-file\nSTORE__QDRANT_COLLECTION=test_coll\n", encoding="utf-8")
    s = Settings(_env_file=env_file)
    assert s.llm.model == "from-file"
    assert s.store.qdrant_collection == "test_coll"


def test_env_var_beats_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM__MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("LLM__MODEL", "from-env")
    assert Settings(_env_file=env_file).llm.model == "from-env"


def test_secrets_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__API_KEY", "sk-super-secret")
    monkeypatch.setenv("STORE__POSTGRES_DSN", "postgresql://u:db-password@h:5432/d")
    s = Settings(_env_file=None)
    text = repr(s) + str(s.model_dump())
    assert "sk-super-secret" not in text
    assert "db-password" not in text
    # 需要时仍可取出明文
    assert s.llm.api_key.get_secret_value() == "sk-super-secret"


def test_invalid_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING__PROVIDER", "unknown")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RERANKER__DEVICE", "tpu")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_openai_compat_requires_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING__PROVIDER", "openai_compat")
    monkeypatch.setenv("EMBEDDING__API_BASE", "")  # 空值等同于未配置
    with pytest.raises(ValidationError, match="API_BASE"):
        Settings(_env_file=None)

    monkeypatch.setenv("EMBEDDING__API_BASE", "https://emb.example.com/v1")
    assert Settings(_env_file=None).embedding.api_base == "https://emb.example.com/v1"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_embedding_model_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # 换工作目录，验证相对路径以项目根目录为基准
    monkeypatch.setenv("EMBEDDING__MODEL_PATH", str(tmp_path / "missing"))
    assert Settings(_env_file=None).embedding.model_source == "BAAI/bge-m3"  # 目录不存在 -> 用模型名
    (tmp_path / "bge").mkdir()
    monkeypatch.setenv("EMBEDDING__MODEL_PATH", str(tmp_path / "bge"))
    assert Settings(_env_file=None).embedding.model_source == str(tmp_path / "bge")
    monkeypatch.setenv("EMBEDDING__MODEL_PATH", "")  # 空值 -> 不使用本地目录
    assert Settings(_env_file=None).embedding.model_source == "BAAI/bge-m3"


def test_collection_name() -> None:
    assert Settings(_env_file=None).collection_name() == "report_chunks__v0_fixed_c500_o50__bge-m3"


def test_goldset_registry() -> None:
    s = Settings(_env_file=None)
    assert s.eval.goldset_file("v1") == PROJECT_ROOT / "eval" / "goldset.jsonl"
    assert set(s.eval.goldsets) >= {"v1", "v2"}


def test_goldset_version_must_be_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL__GOLDSET_VERSION", "v9")
    with pytest.raises(ValidationError, match="v9"):
        Settings(_env_file=None)


def test_frozen_goldset_files_match_registered_sha() -> None:
    """已登记的评测集文件必须与冻结时的 sha256 一致（防止有人修改了已冻结的评测集）。"""
    import hashlib

    s = Settings(_env_file=None)
    for version, frozen in s.eval.goldsets.items():
        assert hashlib.sha256(s.eval.goldset_file(version).read_bytes()).hexdigest() == frozen.sha256, version
