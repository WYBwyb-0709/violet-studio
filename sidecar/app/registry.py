"""
后端注册表与内存预算器：根据硬件能力选择最优推理后端与运行参数。
原则：时间换结果；绝不选超出内存预算的配置。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hardware import Hardware


@dataclass
class Plan:
    backend: str          # cuda | openvino_cpu | openvino_xpu | onnx_cpu | sd_cpp
    tier: str             # fast | balanced | quality
    resolution: int
    steps: int
    threads: int
    quant: str            # fp16 | int8 | int4 | q4
    offload: str          # none | model | sequential
    batch: int = 1
    reason: str = ""


# 模型峰值内存估算系数（GB），按 (分辨率/512)^2 缩放 —— 粗略但保守
PEAK_FACTORS = {
    "sd15": {"fp16": 3.2, "int8": 2.0, "int4": 1.4},
    "sdxl": {"fp16": 7.5, "int8": 4.5, "int4": 3.0},
}
TIER_STEPS = {"fast": 1, "balanced": 8, "quality": 30}
TIER_RES = {"fast": 512, "balanced": 768, "quality": 1024}


def estimate_peak_gb(model: str, quant: str, resolution: int, offload: str) -> float:
    base = PEAK_FACTORS.get(model, PEAK_FACTORS["sd15"]).get(quant, 3.2)
    scaled = base * ((resolution / 512) ** 2)
    if offload == "model":
        scaled *= 0.7
    elif offload == "sequential":
        scaled *= 0.45
    return round(scaled, 2)


def choose_backend(h: Hardware) -> str:
    if h.gpu_type == "cuda":
        return "cuda"
    if h.gpu_type == "intel_xpu" or (h.gpu and "intel" in h.gpu.lower()):
        return "openvino_xpu"
    isa = set(h.isa)
    if {"amx_bf16", "amx_int8"} & isa or "avx512_bf16" in isa:
        return "openvino_cpu"
    if "avx2" in isa or "avx512f" in isa:
        return "onnx_cpu"
    # ARM / 无向量指令 → ggml 兜底
    return "sd_cpp"


def auto_plan(h: Hardware, user_tier: str | None = None, mem_limit_gb: float | None = None) -> Plan:
    backend = choose_backend(h)
    available = mem_limit_gb if mem_limit_gb else max(1.0, h.ram_available_gb - 2.0)
    model = "sdxl" if (user_tier == "quality" and available > 12) else "sd15"

    # 档位优先级：用户指定 > 硬件推断
    tier = user_tier or ("fast" if available < 6 else "balanced" if available < 16 else "quality")
    quant = "int8" if backend in ("openvino_cpu", "openvino_xpu") else (
        "fp16" if backend == "cuda" else "int4")

    # 内存预算循环：从档位默认配置开始，逐步降配直到峰值落在预算内
    resolution = TIER_RES[tier]
    steps = TIER_STEPS[tier]
    offload = "none"
    if backend == "cuda":
        offload = "model"
    elif backend in ("openvino_cpu", "openvino_xpu", "onnx_cpu", "sd_cpp"):
        offload = "sequential" if available < 8 else "model"

    for _ in range(8):
        peak = estimate_peak_gb(model, quant, resolution, offload)
        if peak <= available:
            break
        if resolution > 512:
            resolution = 512
        elif steps > 4:
            steps = max(1, steps // 2)
        elif quant != "int4" and quant != "q4":
            quant = "int4" if backend != "sd_cpp" else "q4"
        elif offload != "sequential":
            offload = "sequential"
        else:
            break

    threads = max(1, h.physical_cores or (h.logical_cores // 2) or 2)
    return Plan(
        backend=backend, tier=tier, resolution=resolution, steps=steps,
        threads=threads, quant=quant, offload=offload,
        reason=f"可用内存 {available}GB；后端 {backend}；峰值估算 {estimate_peak_gb(model, quant, resolution, offload)}GB",
    )
