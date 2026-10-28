"""Tests for POST /validate/alert.

Covers: the happy-path wiring (decode -> call GeminiValidator.validate_alert_with_video
-> response), the >25MB payload -> 413 guard, and malformed Gemini JSON -> 500
(never a 200 with garbage data).
"""
import base64
import os
import sys

# Settings() is instantiated at import time by src.app.settings and requires
# these env vars; set harmless test defaults before any app import so the
# import doesn't blow up when a real .env isn't present in this environment.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("NVIDIA_NIM_ENDPOINT", "http://localhost:0")
os.environ.setdefault("NVIDIA_NIM_API_KEY", "test-nvidia-key")
os.environ.setdefault("VALIDATOR_API_KEY", "test-validator-key")

# src/app/validators/nvidia_validator.py does `from instructor.retry import
# InstructorRetryException`, a path removed in newer `instructor` releases
# (now under `instructor.core.retry`). This is a pre-existing environment/
# dependency-version mismatch unrelated to this feature (nvidia_validator.py
# is out of scope for this task), so shim the old module path here rather
# than touching that file.
if "instructor.retry" not in sys.modules:
    try:
        import instructor.retry  # noqa: F401
    except ModuleNotFoundError:
        import instructor.core.retry as _instructor_core_retry

        sys.modules["instructor.retry"] = _instructor_core_retry

from fastapi.testclient import TestClient

from main import app
from src.app.settings import settings
from src.app.routers import validate as validate_router

client = TestClient(app)

AUTH_HEADERS = {"X-API-Key": settings.validator_api_key}


def _video_b64(size_bytes: int = 16) -> str:
    return base64.b64encode(b"\x00" * size_bytes).decode("utf-8")


def test_validate_alert_happy_path(monkeypatch):
    """Decode -> call -> response should wire through exactly, confirming the
    endpoint passes the right args and maps the validator result 1:1 into
    AlertValidationResponse.
    """
    captured_args = {}

    def fake_validate_alert_with_video(video_bytes, detection_label, detection_metadata, system_prompt):
        captured_args["video_bytes"] = video_bytes
        captured_args["detection_label"] = detection_label
        captured_args["detection_metadata"] = detection_metadata
        captured_args["system_prompt"] = system_prompt
        return {
            "success": True,
            "confirmed": True,
            "confidence": 87,
            "severity": "HIGH",
            "reasoning": "Person clearly running away from the scene after the alarm.",
        }

    monkeypatch.setattr(
        validate_router._gemini_validator,
        "validate_alert_with_video",
        fake_validate_alert_with_video,
    )

    payload = {
        "video_base64": _video_b64(),
        "detection_label": "Person Running",
        "detection_metadata": {"zone": "loading_dock", "rule_id": "rule-42"},
        "system_prompt": "Flag any unauthorized access to the loading dock.",
    }

    response = client.post("/validate/alert", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "confirmed": True,
        "confidence": 87,
        "severity": "HIGH",
        "reasoning": "Person clearly running away from the scene after the alarm.",
    }

    assert captured_args["video_bytes"] == base64.b64decode(payload["video_base64"])
    assert captured_args["detection_label"] == "Person Running"
    assert captured_args["detection_metadata"] == {"zone": "loading_dock", "rule_id": "rule-42"}
    assert captured_args["system_prompt"] == "Flag any unauthorized access to the loading dock."


def test_validate_alert_rejects_oversized_video(monkeypatch):
    """A decoded video payload over 25MB must be rejected with 413, before any
    call reaches the validator.
    """
    called = {"count": 0}

    def fake_validate_alert_with_video(*args, **kwargs):
        called["count"] += 1
        return {"success": True, "confirmed": True, "confidence": 50, "severity": "LOW", "reasoning": "x"}

    monkeypatch.setattr(
        validate_router._gemini_validator,
        "validate_alert_with_video",
        fake_validate_alert_with_video,
    )

    oversized_bytes = b"\x00" * (25 * 1024 * 1024 + 1)
    payload = {
        "video_base64": base64.b64encode(oversized_bytes).decode("utf-8"),
        "detection_label": "Person Running",
        "detection_metadata": None,
        "system_prompt": "",
    }

    response = client.post("/validate/alert", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 413
    assert called["count"] == 0


def test_validate_alert_malformed_gemini_response_returns_500(monkeypatch):
    """When the validator can't parse Gemini's output (malformed JSON, missing
    fields, etc.) the endpoint must surface a 500, never a 200 with garbage.
    """

    def fake_validate_alert_with_video(video_bytes, detection_label, detection_metadata, system_prompt):
        return {"success": False, "error": "Invalid JSON from Gemini: not json at all"}

    monkeypatch.setattr(
        validate_router._gemini_validator,
        "validate_alert_with_video",
        fake_validate_alert_with_video,
    )

    payload = {
        "video_base64": _video_b64(),
        "detection_label": "Person Running",
        "detection_metadata": None,
        "system_prompt": "",
    }

    response = client.post("/validate/alert", json=payload, headers=AUTH_HEADERS)

    assert response.status_code == 500
    assert "Invalid JSON from Gemini" in response.json()["detail"]
