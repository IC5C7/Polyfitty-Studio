from __future__ import annotations

from hidden_entry import main
from startup_gpu_preflight import _run_startup_gpu_preflight

def _verify_portable_features() -> int:
    from runtime_contract import verify_runtime_manifest
    verify_runtime_manifest(require_source_match=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
