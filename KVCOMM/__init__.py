from __future__ import annotations

import os
from pkgutil import extend_path

if os.environ.get("KVCOMM_EXTEND_PATH", "0").lower() in {"1", "true", "yes", "y"}:
    __path__ = extend_path(__path__, __name__)
