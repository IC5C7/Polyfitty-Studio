from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SUFFIXES = {
    ".py",
    ".ps1",
    ".psm1",
    ".psd1",
    ".bat",
    ".cmd",
    ".vbs",
    ".hta",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".spec",
    ".c",
    ".cpp",
    ".cs",
    ".h",
    ".hpp",
    ".cmake",
}
EXCLUDED_PARTS = {".git", ".build", "build", "dist", "__pycache__", ".venv", "venv"}

def _read_script(path: Path) -> str:
    payload = path.read_bytes()
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")
    return payload.decode("utf-8-sig")

def _is_hangul(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0xD7B0 <= codepoint <= 0xD7FF
    )

def test_all_script_sources_use_english_not_korean():
    violations = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        source = _read_script(path)
        for line_number, line in enumerate(source.splitlines(), start=1):
            if any(_is_hangul(character) for character in line):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}")

    assert not violations, "Script sources must be English-only:\n" + "\n".join(violations)
