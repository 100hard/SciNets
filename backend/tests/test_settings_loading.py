from __future__ import annotations

import importlib


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TIER2_LLM_MODEL", "gpt-test")

    config_module = importlib.import_module("backend.app.core.config")
    Settings = config_module.Settings

    settings = Settings()

    assert settings.openai_api_key == "test-key"
    assert settings.tier2_llm_model == "gpt-test"
