"""Optional immutable-revision and artifact-integrity helpers for model loading.

Integrity checks are opt-in. If no revision or digest map is supplied, existing
local-model and Hugging Face workflows retain their historical behavior.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Mapping, Optional

# Reviewed Hugging Face commits for the three published checkpoint families.
# The bundled repository uses one commit for all three subfolders.
MODEL_REVISIONS = {
    "convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "convaiinnovations/laya-multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    "convaiinnovations/laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
}


def default_revision(repo: str, _subfolder: Optional[str] = None) -> Optional[str]:
    """Return the reviewed revision for a published model, or None for custom repos."""
    if repo == "convaiinnovations/laya":
        return MODEL_REVISIONS[repo]
    return MODEL_REVISIONS.get(repo)


def _digest_map(value: Optional[Mapping[str, str]]) -> dict[str, str]:
    if value is None:
        raw = os.environ.get("LAYA_SHA256_DIGESTS", "").strip()
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("LAYA_SHA256_DIGESTS must be a JSON object of artifact->sha256") from exc
    if not isinstance(value, Mapping):
        raise ValueError("artifact digests must be a mapping of artifact names to SHA-256 strings")
    result: dict[str, str] = {}
    for key, expected in value.items():
        key = str(key).strip().replace("\\", "/")
        expected = str(expected).strip().lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("SHA-256 digest for %r must contain exactly 64 hexadecimal characters" % key)
        result[key] = expected
    return result


def _artifact_path(model_dir: str, name: str, onnx_path: Optional[str] = None) -> str:
    aliases = {
        "config": "rl_agent_config.json",
        "weights": "model.safetensors",
        "tokenizer": "tokenizer/tokenizer.json",
        "onnx": onnx_path or "laya.onnx",
        "encoder": "encoder.onnx",
        "head": "head.onnx",
    }
    return os.path.join(model_dir, aliases.get(name, name)) if name not in ("onnx",) or onnx_path is None else onnx_path


def verify_file(path: str, expected: str) -> None:
    """Verify one file or raise before any model/runtime parser receives it."""
    if not os.path.isfile(path):
        raise FileNotFoundError("integrity artifact not found: %s" % path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ValueError("SHA-256 mismatch for %s: expected %s, got %s" % (path, expected, actual))


def verify_artifacts(
    model_dir: str,
    digests: Optional[Mapping[str, str]] = None,
    onnx_path: Optional[str] = None,
) -> None:
    """Verify configured artifacts; absent configuration is a no-op."""
    for name, expected in _digest_map(digests).items():
        path = onnx_path if name == "onnx" and onnx_path else _artifact_path(model_dir, name, onnx_path)
        verify_file(path, expected)
