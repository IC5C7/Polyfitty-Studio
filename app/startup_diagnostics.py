from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

def startup_trace(message: str) -> None:
    """Append a lightweight startup milestone when launched by the installer."""
    target = os.environ.get("POLYFITTY_STARTUP_TRACE", os.environ.get("POLYFIT_STARTUP_TRACE", "")).strip()
    if not target:
        return
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"{stamp} | {message}\n")
    except Exception:

        pass
