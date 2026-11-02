"""Offline tests for the Ollama brain provider (no network, no server).

Run with:  pytest tests/test_ollama.py
"""

import asyncio

import pytest

from harvey.config import BrainConfig
from harvey.ollama import (
    DEFAULT_BASE_URL,
    OllamaClient,
    OllamaError,
    OllamaQuotaClient,
    OllamaResult,
)


# ── Config ──────────────────────────────────────────────────────────────

def test_ollama_is_a_valid_provider():
    cfg = BrainConfig(provider="ollama", model="qwen2.5:14b")
    assert cfg.provider == "ollama"
    assert cfg.model == "qwen2.5:14b"


def test_ollama_default_model_blank_is_allowed():
    cfg = BrainConfig(provider="ollama")
    assert cfg.model == ""


def test_ollama_base_url_default_blank():
    cfg = BrainConfig(provider="ollama")
    assert cfg.base_url == ""


def test_ollama_base_url_accepts_valid():
    cfg = BrainConfig(provider="ollama", base_url="http://10.0.0.5:11434")
    assert cfg.base_url == "http://10.0.0.5:11434"


def test_ollama_base_url_rejects_schemeless():
    with pytest.raises(Exception):
        BrainConfig(provider="ollama", base_url="127.0.0.1:11434")


def test_unknown_provider_still_rejected():
    with pytest.raises(Exception):
        BrainConfig(provider="lmstudio")


# ── Client construction ─────────────────────────────────────────────────

def test_client_defaults():
    c = OllamaClient()
    assert c.base_url == DEFAULT_BASE_URL
    assert c.model  # suggested model, non-empty


def test_client_explicit_model_and_url():
    c = OllamaClient(model="llama3.1:8b", base_url="http://192.168.1.20:11434/")
    assert c.model == "llama3.1:8b"
    assert c.base_url == "http://192.168.1.20:11434"  # trailing slash stripped


def test_client_base_url_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://172.16.0.9:11434")
    c = OllamaClient()
    assert c.base_url == "http://172.16.0.9:11434"


def test_client_env_override_ignored_when_url_given(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://172.16.0.9:11434")
    c = OllamaClient(base_url="http://10.1.1.1:11434")
    assert c.base_url == "http://10.1.1.1:11434"


def test_client_rejects_schemeless_url():
    with pytest.raises(OllamaError):
        OllamaClient(base_url="localhost:11434")


def test_fingerprint_is_endpoint_and_model():
    c = OllamaClient(model="m1", base_url="http://a:1")
    assert c.fingerprint() == ("http://a:1", "m1")


# ── Error taxonomy ──────────────────────────────────────────────────────

def test_error_kinds_and_retryability():
    e1 = OllamaError("x", kind="setup", retryable=False)
    assert e1.kind == "setup" and e1.retryable is False
    e2 = OllamaError("y", kind="runtime", retryable=True)
    assert e2.kind == "runtime" and e2.retryable is True


def test_result_defaults():
    r = OllamaResult(text="hi")
    assert r.text == "hi"
    assert r.usage == {}
    assert r.duration_ms == 0


# ── Quota client ────────────────────────────────────────────────────────

def test_quota_client_reports_no_utilization():
    async def _run():
        q = OllamaQuotaClient(OllamaClient())
        assert await q.get_utilization() is None
        q.invalidate()  # interface parity; must not raise

    asyncio.run(_run())


# ── Brain wiring (no network: preflight of a dead endpoint must be a
#    clean pause reason, not a crash) ────────────────────────────────────

def test_brain_wires_ollama_provider(tmp_path):
    from harvey.state import StateManager
    from harvey.brain import Brain

    state = StateManager(db_path=str(tmp_path / "t.db"))
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen2.5:14b"))
    assert brain.provider == "ollama"
    assert brain.provider_label == "Ollama"
    assert brain.ollama is not None
    assert brain.quota is not None
    # Claude/Codex clients must not be constructed for this provider.
    assert brain.codex is None


def test_brain_ollama_preflight_dead_endpoint_is_a_pause_reason(tmp_path):
    from harvey.state import StateManager
    from harvey.brain import Brain

    state = StateManager(db_path=str(tmp_path / "t.db"))
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen2.5:14b",
                                     base_url="http://127.0.0.1:1"))
    reason = asyncio.run(brain.preflight())
    assert reason  # non-empty: the server is not there
    assert "127.0.0.1:1" in reason


def test_brain_ollama_think_returns_empty_when_paused(tmp_path):
    from harvey.state import StateManager
    from harvey.brain import Brain

    state = StateManager(db_path=str(tmp_path / "t.db"))
    brain = Brain(state, BrainConfig(provider="ollama", model="qwen2.5:14b",
                                     base_url="http://127.0.0.1:1"))
    # Mark it unavailable (as preflight would), then a call must skip,
    # not crash, and return "".
    asyncio.run(brain._mark_unavailable(OllamaError("server down", kind="setup")))
    out = asyncio.run(brain.think("hello", max_retries=0))
    assert out == ""


# ── Usage accounting plumbing (state layer) ─────────────────────────────

def test_usage_log_column_mapping_includes_ollama():
    from harvey.state import _USAGE_LOG_COLUMN
    assert _USAGE_LOG_COLUMN["ollama"] == "ollama_calls"


def test_state_increment_and_get_ollama_usage(tmp_path):
    from harvey.state import StateManager

    async def _run():
        state = StateManager(db_path=str(tmp_path / "t.db"))
        await state.init_db()
        await state.increment_usage(provider="ollama")
        await state.increment_usage(provider="ollama")
        assert await state.get_usage_today(provider="ollama") == 2
        # Other providers untouched.
        assert await state.get_usage_today(provider="claude") == 0
        assert await state.get_usage_today(provider="codex") == 0

    asyncio.run(_run())