"""Test package bootstrap.

Point the engine's app-support dir (logs, library.db, …) at a throwaway
temp dir for the whole test session unless the environment already set
one. Without this, ``append_app_log`` and ``database()`` calls made by
tests land in the *real* ``~/Library/Application Support/EkoVideo
Compressor`` on a developer's machine — polluting the user's actual
app.log with test fixtures. Per-test ``patch.dict`` overrides still
take precedence.
"""

from __future__ import annotations

import os
import tempfile

if not os.environ.get("EKO_APP_SUPPORT_DIR"):
    os.environ["EKO_APP_SUPPORT_DIR"] = tempfile.mkdtemp(prefix="ekovideo-tests-")
