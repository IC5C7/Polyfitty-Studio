from __future__ import annotations

"""Best-effort GPU telemetry helpers for the Performance panel.

Telemetry is deliberately optional: OpenCL itself does not expose a portable
utilization counter. NVIDIA uses nvidia-smi when present. Windows AMD (and
other vendors) use the OS GPU Engine / GPU Process Memory performance counters.
Unavailable counters are reported as N/A rather than fabricated.
"""

import csv
import io
import json
import os
import platform
import shutil
import subprocess
from typing import Any

def _empty(detail: str, source: str = "unavailable") -> dict[str, Any]:
    return {
        "utilization_percent": None,
        "memory_used_bytes": None,
        "memory_total_bytes": None,
        "source": source,
        "detail": str(detail),
    }

def _find_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if os.name == "nt":
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
            os.path.expandvars(r"%ProgramW6432%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
    return None

def _query_nvidia_smi(device_name: str) -> dict[str, Any]:
    executable = _find_nvidia_smi()
    if not executable:
        return _empty("nvidia-smi is not installed or is not on PATH.")
    command = [
        executable,
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=3,
            check=False, creationflags=creationflags,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "nvidia-smi query failed").strip()
            return _empty(detail[:500], "nvidia-smi")
        wanted = "".join(ch.lower() for ch in str(device_name) if ch.isalnum())
        rows = list(csv.reader(io.StringIO(completed.stdout)))
        for row in rows:
            if len(row) < 4:
                continue
            name = row[0].strip()
            normalized = "".join(ch.lower() for ch in name if ch.isalnum())
            if wanted and wanted not in normalized and normalized not in wanted:
                continue
            utilization = None if row[1].strip().upper() in {"N/A", "[N/A]"} else float(row[1].strip())
            memory_used = None if row[2].strip().upper() in {"N/A", "[N/A]"} else int(float(row[2].strip()) * 1024 * 1024)
            memory_total = None if row[3].strip().upper() in {"N/A", "[N/A]"} else int(float(row[3].strip()) * 1024 * 1024)
            return {
                "utilization_percent": utilization,
                "memory_used_bytes": memory_used,
                "memory_total_bytes": memory_total,
                "source": "NVIDIA-SMI device telemetry",
                "detail": f"Device-level telemetry for {name}.",
            }
        return _empty(f"The selected NVIDIA device was not found in nvidia-smi output: {device_name}", "nvidia-smi")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return _empty(f"nvidia-smi telemetry unavailable ({type(exc).__name__}: {exc})", "nvidia-smi")

def _query_windows_gpu_counters(pid: int) -> dict[str, Any]:
    """Read per-process GPU engine and dedicated-memory counters on Windows."""
    if os.name != "nt":
        return _empty("Windows GPU performance counters are not available on this platform.")
    pid_value = max(0, int(pid))
    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
$pidToken = 'pid_' + $PID_TO_QUERY + '_'
$load = $null
$memory = $null
try {
  $samples = Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage' -MaxSamples 1 -ErrorAction Stop
  $values = @($samples.CounterSamples | Where-Object { $_.InstanceName -like ('*' + $pidToken + '*') } | ForEach-Object { [double]$_.CookedValue })
  if ($values.Count -gt 0) { $load = [math]::Min(100.0, [math]::Max(0.0, (($values | Measure-Object -Maximum).Maximum))) }
  else { $load = 0.0 }
} catch { $load = $null }
try {
  $memorySamples = Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage' -MaxSamples 1 -ErrorAction Stop
  $memoryValues = @($memorySamples.CounterSamples | Where-Object { $_.InstanceName -like ('*' + $pidToken + '*') } | ForEach-Object { [double]$_.CookedValue })
  if ($memoryValues.Count -gt 0) { $memory = [int64](($memoryValues | Measure-Object -Sum).Sum) }
} catch { $memory = $null }
[pscustomobject]@{ utilization_percent = $load; memory_used_bytes = $memory; source = 'Windows GPU Engine / Process Memory counters' } | ConvertTo-Json -Compress
""".replace("$PID_TO_QUERY", str(pid_value))
    command = [
        "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
        "-ExecutionPolicy", "Bypass", "-Command", script,
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5,
            check=False, creationflags=creationflags,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Windows GPU counters query failed").strip()
            return _empty(detail[:500], "Windows GPU counters")
        json_line = next((line.strip() for line in reversed(completed.stdout.splitlines()) if line.strip().startswith("{")), "")
        if not json_line:
            return _empty("Windows GPU counters returned no sample. Check that GPU Engine counters are enabled.", "Windows GPU counters")
        payload = json.loads(json_line)
        utilization = payload.get("utilization_percent")
        memory_used = payload.get("memory_used_bytes")
        return {
            "utilization_percent": None if utilization is None else max(0.0, min(100.0, float(utilization))),
            "memory_used_bytes": None if memory_used is None else max(0, int(memory_used)),
            "memory_total_bytes": None,
            "source": str(payload.get("source") or "Windows GPU counters"),
            "detail": "Peak per-process GPU engine utilization. This is not a whole-device aggregate.",
        }
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        return _empty(f"Windows GPU telemetry unavailable ({type(exc).__name__}: {exc})", "Windows GPU counters")

def query_gpu_telemetry(vendor: str, device_name: str, pid: int | None = None) -> dict[str, Any]:
    """Query available driver/OS telemetry; never estimates utilization."""
    normalized_vendor = str(vendor or "").lower()
    if "nvidia" in normalized_vendor:
        result = _query_nvidia_smi(device_name)
        if result.get("utilization_percent") is not None or platform.system() != "Windows":
            return result

    if platform.system() == "Windows":
        return _query_windows_gpu_counters(os.getpid() if pid is None else int(pid))
    return _empty("No supported utilization telemetry provider was found for this GPU/OS.")
