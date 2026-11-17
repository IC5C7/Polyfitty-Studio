from __future__ import annotations

"""Create or verify the source identity shared by installed and portable builds."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from runtime_contract import verify_source_runtime_manifest, write_runtime_manifest

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="Create runtime_manifest.json for this exact source tree.")
    action.add_argument("--verify", action="store_true", help="Verify the source tree against runtime_manifest.json.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing app/ and VERSION.txt.")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if args.write:
        manifest_path = write_runtime_manifest(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"Runtime source manifest created: Polyfitty Studio {manifest['app_version']} | "
            f"{manifest['source_fingerprint']} | {manifest['source_file_count']} files"
        )
        print(manifest_path)
        return 0

    manifest = verify_source_runtime_manifest(root)
    print(
        f"Runtime source manifest verified: Polyfitty Studio {manifest['app_version']} | "
        f"{manifest['source_fingerprint']} | {manifest['source_file_count']} files"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
