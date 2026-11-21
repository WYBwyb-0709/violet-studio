"""
硬件探测：CPU 型号、物理/逻辑核心、 ISA 指令集、内存、GPU/加速器。
跨平台：Windows（wmic / platform / psutil）、Linux（/proc/cpuinfo, sysfs）。
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Hardware:
    os: str = ""
    arch: str = ""
    cpu: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    isa: list[str] = field(default_factory=list)
    gpu: str | None = None
    gpu_type: str | None = None        # cuda | intel_xpu | rocm | none
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_free_gb: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return ""


def _detect_isa_linux() -> list[str]:
    flags = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            flags = f.read()
    except Exception:
        pass
    m = re.search(r"flags\s*:\s*(.+)", flags)
    tokens = set(m.group(1).split()) if m else set()
    wanted = ["avx", "avx2", "avx512f", "avx512bw", "avx512_vnni", "avx512_bf16",
              "amx_bf16", "amx_int8", "amx_tile", "sse4_2", "fma"]
    found = [w for w in wanted if w in tokens]
    # ARM
    if "asimd" in tokens or "neon" in tokens:
        found.append("neon")
    if "sve" in tokens:
        found.append("sve")
    return found


def _detect_isa_windows() -> list[str]:
    # Windows 无直接 flags；用 PowerShell 读取注册表 + 已知 CPU 名推断，保守返回。
    out = _run(["powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor).Name"])
    name = out.strip().lower()
    found = []
    if name:
        found.append("sse4_2")
        if any(k in name for k in ("core", "xeon", "ryzen", "epyc", "threadripper")):
            found.append("avx")
            found.append("avx2")
        if any(k in name for k in ("ice lake", "icelake", "granite rapids", "sapphire rapids",
                                   "emerald rapids", "zen 4", "zen4", "arrow lake", "meteor lake")):
            found += ["avx512f", "avx512bw"]
        if "granite rapids" in name or "sapphire rapids" in name:
            found += ["amx_bf16", "amx_int8", "amx_tile"]
    return found


def detect() -> Hardware:
    h = Hardware()
    h.os = platform.system()
    h.arch = platform.machine().lower()

    try:
        import psutil
        h.logical_cores = psutil.cpu_count(logical=True) or 0
        h.physical_cores = psutil.cpu_count(logical=False) or 0
        mem = psutil.virtual_memory()
        h.ram_total_gb = round(mem.total / 1024**3, 1)
        h.ram_available_gb = round(mem.available / 1024**3, 1)
        h.disk_free_gb = round(psutil.disk_usage(os.getcwd()).free / 1024**3, 1)
        h.cpu = platform.processor() or ""
    except Exception:
        h.logical_cores = os.cpu_count() or 0
        h.physical_cores = h.logical_cores

    if h.os == "Linux":
        h.isa = _detect_isa_linux()
        # CPU 型号从 /proc/cpuinfo 读取：platform.processor() 在 Linux 常返回架构名（如 x86_64）或空
        if not h.cpu or h.cpu in ("x86_64", "aarch64", "amd64", ""):
            out = _run(["grep", "-m1", "model name", "/proc/cpuinfo"])
            if ":" in out:
                h.cpu = out.split(":", 1)[1].strip()
    elif h.os == "Windows":
        h.isa = _detect_isa_windows()
        if not h.cpu:
            h.cpu = _run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor).Name"]).strip()
    else:
        h.isa = []

    # GPU / 加速器探测
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if out.strip():
            h.gpu = out.strip().splitlines()[0]
            h.gpu_type = "cuda"
    if not h.gpu and os.path.exists("/dev/dri"):
        h.gpu = "Intel/AMD DRM device (/dev/dri)"
        h.gpu_type = "intel_xpu" if h.os == "Linux" else None
    return h
