from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

try:
    import certifi
except Exception:  # pragma: no cover - fallback for minimal smoke-test envs
    certifi = None


HF_API_BASE = "https://huggingface.co"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"
HF_GATED_MODEL_CHECKS: list[tuple[str, str, str]] = [
    ("pyannote/segmentation-3.0", "config.yaml", "Segmentation pyannote 3.0"),
    ("pyannote/speaker-diarization-3.1", "config.yaml", "Diarisation pyannote 3.1"),
    ("pyannote/speaker-diarization-community-1", "config.yaml", "Diarisation Community-1"),
]


def _hf_request(url: str, token: str = "", method: str = "GET") -> urllib.request.Request:
    headers = {"User-Agent": "EkoVideoCompressor/engine"}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    return urllib.request.Request(url, headers=headers, method=method)


def _ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def hf_whoami(token: str) -> dict:
    with urllib.request.urlopen(
        _hf_request(f"{HF_API_BASE}/api/whoami-v2", token),
        timeout=12,
        context=_ssl_context(),
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def hf_file_access_status(token: str, repo_id: str, filename: str) -> tuple[bool, str]:
    url = f"{HF_API_BASE}/{repo_id}/resolve/main/{filename}"
    try:
        with urllib.request.urlopen(
            _hf_request(url, token, method="HEAD"),
            timeout=12,
            context=_ssl_context(),
        ) as response:
            return 200 <= response.status < 400, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return False, "license not accepted or token has no access"
        if exc.code == 404:
            return False, "control file not found"
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def hf_check(token: str) -> dict:
    account = hf_whoami(token) if token.strip() else {}
    checks = []
    for repo_id, filename, label in HF_GATED_MODEL_CHECKS:
        ok, detail = hf_file_access_status(token, repo_id, filename)
        checks.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "label": label,
                "ok": ok,
                "detail": detail,
            }
        )
    return {"account": account, "checks": checks}
