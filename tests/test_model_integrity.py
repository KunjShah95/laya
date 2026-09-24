"""Tests for optional model revision and SHA-256 integrity controls."""
import hashlib
import json

import pytest

from laya.model_integrity import MODEL_REVISIONS, default_revision, verify_artifacts, verify_file


def test_published_models_have_reviewed_revisions():
    assert MODEL_REVISIONS["convaiinnovations/laya"]
    assert default_revision("convaiinnovations/laya", "multilingual") == MODEL_REVISIONS["convaiinnovations/laya"]
    assert default_revision("custom/model") is None


def test_router_passes_published_revision_to_agent(monkeypatch):
    import laya.agent as agent_module
    from laya.router import Router

    calls = []

    class FakeAgent:
        def __init__(self, repo, **kwargs):
            calls.append((repo, kwargs))

    monkeypatch.setattr(agent_module, "Agent", FakeAgent)
    Router().load("multilingual")
    assert calls[0][0] == "convaiinnovations/laya"
    assert calls[0][1]["revision"] == MODEL_REVISIONS["convaiinnovations/laya"]
    assert calls[0][1]["subfolder"] == "multilingual"


def test_verify_file_accepts_matching_digest_and_rejects_mismatch(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"reviewed model bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    verify_file(str(artifact), digest)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(str(artifact), "0" * 64)


def test_digest_map_can_be_loaded_from_environment(tmp_path, monkeypatch):
    artifact = tmp_path / "rl_agent_config.json"
    artifact.write_text(json.dumps({"ok": True}))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setenv("LAYA_SHA256_DIGESTS", json.dumps({"config": digest}))
    verify_artifacts(str(tmp_path))
    monkeypatch.setenv("LAYA_SHA256_DIGESTS", json.dumps({"config": "0" * 64}))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_artifacts(str(tmp_path))
