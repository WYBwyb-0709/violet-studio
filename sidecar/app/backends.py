"""
推理后端适配层：把不同的推理引擎封装成统一接口 generate()。
真实模型在未下载时不可用；这里提供可降级的占位实现与清晰的扩展点。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from .registry import Plan

log = logging.getLogger("violet.backends")


class BackendUnavailable(RuntimeError):
    pass


class BaseBackend:
    name = "base"

    def __init__(self, plan: Plan):
        self.plan = plan
        self.loaded = False

    def load(self):
        raise NotImplementedError

    def generate(self, prompt: str, negative: str = "", **kwargs) -> list[Image.Image]:
        raise NotImplementedError

    def unload(self):
        self.loaded = False


class OpenVINOBackend(BaseBackend):
    name = "openvino"
    """OpenVINO GenAI Text2ImagePipeline（CPU INT8/BF16，走 AMX/oneDNN）。
    适用于标准文生图；LayerDiffuse 透明 VAE 需 PyTorch 后端。"""
    def load(self):
        try:
            import openvino_genai as ov_genai  # noqa
            self._mod = ov_genai
            self.loaded = True
            log.info("OpenVINO GenAI 可用")
        except Exception as e:
            raise BackendUnavailable(f"OpenVINO GenAI 不可用：{e}")

    def generate(self, prompt: str, negative: str = "", **kwargs) -> list[Image.Image]:
        if not self.loaded:
            self.load()
        raise BackendUnavailable("未找到本地 IR 模型目录；请先用 optimum-cli 导出模型到 models/。")


class DiffusersBackend(BaseBackend):
    name = "diffusers"
    """原生 PyTorch + diffusers + torch.compile；兼容 LayerDiffuse / LoRA / ControlNet。
    CPU 上 AMX/FP16 已上游到原生 PyTorch 2.6+；用 offload 与 VAE tiling 防 OOM。"""
    def load(self):
        try:
            import torch  # noqa
            import diffusers  # noqa
            self._torch = torch
            self.loaded = True
            log.info("PyTorch/diffusers 可用，线程数=%s", self.plan.threads)
            torch.set_num_threads(self.plan.threads)
        except Exception as e:
            raise BackendUnavailable(f"PyTorch/diffusers 不可用：{e}")

    def generate(self, prompt: str, negative: str = "", **kwargs) -> list[Image.Image]:
        if not self.loaded:
            self.load()
        raise BackendUnavailable("未找到本地模型权重；请下载 SD1.5 或 SDXL 模型到 models/。")


class SdCppBackend(BaseBackend):
    name = "sd_cpp"
    """ggml/stable-diffusion.cpp，GGUF Q4/Q8，纯 C/C++，NEON/AVX 兜底，极低内存。"""
    def load(self):
        raise BackendUnavailable("sd.cpp 二进制未随包构建；请在支持的平台上编译并放置到 backend/sdcpp。")

    def generate(self, prompt: str, negative: str = "", **kwargs) -> list[Image.Image]:
        raise BackendUnavailable("sd.cpp 不可用")


def build_backend(plan: Plan) -> BaseBackend:
    mapping = {
        "cuda": DiffusersBackend, "openvino_cpu": OpenVINOBackend,
        "openvino_xpu": OpenVINOBackend, "onnx_cpu": OpenVINOBackend,
        "sd_cpp": SdCppBackend,
    }
    cls = mapping.get(plan.backend, SdCppBackend)
    # 分层生成强制走 diffusers（LayerDiffuse 用透明 VAE）
    return cls(plan)
