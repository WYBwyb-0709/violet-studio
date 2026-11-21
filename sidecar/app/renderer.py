"""
2.5D 渲染器（快路径）：用每层的 albedo + normal + depth，按光照参数实时重打光并按 Z 间距做视差合成。
纯 NumPy/PIL 实现——不调用 AI 模型、不占模型内存、秒级、可高并发。
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class Layer:
    id: Any
    name: str = ""
    rgba: str | None = None
    albedo: str | None = None
    normal: str | None = None
    depth: str | None = None
    z: float = 0.0
    opacity: float = 1.0
    visible: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Layer":
        return cls(
            id=d.get("id", 0), name=d.get("name", ""),
            rgba=d.get("rgba"), albedo=d.get("albedo"),
            normal=d.get("normal"), depth=d.get("depth"),
            z=float(d.get("z_depth", d.get("z", 0.0))),
            opacity=float(d.get("opacity", 1.0)),
            visible=bool(d.get("visible", True)),
        )


@dataclass
class Lighting:
    key_azimuth: float = 45.0
    key_elevation: float = 45.0
    key_intensity: float = 1.0
    fill_intensity: float = 0.4
    ambient: float = 0.6
    ao: float = 0.8

    @classmethod
    def from_dict(cls, d: dict) -> "Lighting":
        key = d.get("key", {}) if isinstance(d, dict) else {}
        return cls(
            key_azimuth=float(key.get("azimuth", 45)),
            key_elevation=float(key.get("elevation", 45)),
            key_intensity=float(key.get("intensity", 1.0)),
            fill_intensity=float(d.get("fill", {}).get("intensity", 0.4)) if isinstance(d, dict) else 0.4,
            ambient=float(d.get("ambient", 0.6)) if isinstance(d, dict) else 0.6,
            ao=float(d.get("ao", 0.8)) if isinstance(d, dict) else 0.8,
        )


def _load(path: str) -> np.ndarray | None:
    if not path:
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None


def _light_dir(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float32)


def relight(albedo: np.ndarray, normal: np.ndarray | None, depth: np.ndarray | None, light: Lighting) -> np.ndarray:
    """Lambert + 简化 Blinn-Phong + AO 的重打光，输出 [0,1] RGB。"""
    h, w, _ = albedo.shape
    if normal is None or normal.shape != albedo.shape:
        # 无 normal：退化为平面法线，仅整体亮度变化（仍可调，但无表面起伏）
        normal = np.zeros_like(albedo); normal[..., 2] = 1.0
    n = normal * 2.0 - 1.0  # [0,1] -> [-1,1]
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)

    L = _light_dir(light.key_azimuth, light.key_elevation)
    diff = np.clip(np.sum(n * L, axis=-1, keepdims=True), 0, 1)
    key = diff * light.key_intensity
    fill = np.clip(np.sum(n * np.array([-0.5, -0.3, 0.6]), axis=-1, keepdims=True), 0, 1) * light.fill_intensity

    ao_map = 1.0
    if depth is not None and depth.shape[:2] == (h, w):
        if depth.ndim == 3:
            depth = depth[..., 0]
        # 用局部梯度近似 AO（凹陷处更暗）
        gx = np.abs(np.gradient(depth, axis=1))
        gy = np.abs(np.gradient(depth, axis=0))
        ao_map = np.clip(1.0 - (gx + gy) * 4.0 * light.ao, 0.3, 1.0)[..., None]

    illum = np.clip(light.ambient * ao_map + key + fill, 0, 1.6)
    return np.clip(albedo * illum, 0, 1)


def render(scene: dict, lighting: dict | None, camera_shift: tuple[float, float] = (0.0, 0.0)) -> Image.Image:
    """按 Z 间距做视差偏移并合成所有可见层。"""
    layers_data = scene.get("layers", [])
    canvas = scene.get("canvas", {})
    w, h = int(canvas.get("width", 512)), int(canvas.get("height", 512))
    light = Lighting.from_dict(lighting or {})

    z_list = [float(l.get("z_depth", l.get("z", 0))) for l in layers_data if l.get("visible", True)]
    z_ref = max(z_list) if z_list else 0.0
    z_min = min(z_list) if z_list else 0.0
    z_span = max(1e-3, z_ref - z_min)

    out = np.zeros((h, w, 3), dtype=np.float32)
    # 由远及近合成（z 越大越靠近观察者）
    ordered = sorted([l for l in layers_data if l.get("visible", True)],
                     key=lambda l: float(l.get("z_depth", l.get("z", 0))))
    for ld in ordered:
        layer = Layer.from_dict(ld)
        src = layer.albedo or layer.rgba
        img = _load(src) if src else None
        if img is None:
            continue
        normal = _load(layer.normal) if layer.normal else None
        depth = _load(layer.depth) if layer.depth else None
        lit = relight(img, normal, depth, light)

        # 视差：z 越大（越近）随相机移动越多
        factor = (layer.z - z_min) / z_span
        dx = int(camera_shift[0] * factor * 20)
        dy = int(camera_shift[1] * factor * 20)
        lit = np.roll(lit, shift=(dy, dx), axis=(0, 1))

        # alpha 遮罩
        alpha = np.ones((h, w, 1), dtype=np.float32) * layer.opacity
        if layer.rgba:
            try:
                rgba = np.asarray(Image.open(layer.rgba).convert("RGBA"), dtype=np.float32)
                if rgba.ndim == 3 and rgba.shape[2] == 4:
                    alpha = (rgba[..., 3:4] / 255.0) * layer.opacity
            except Exception:
                pass
        ih, iw = lit.shape[:2]
        hh, ww = min(h, ih), min(w, iw)
        out[:hh, :ww] = lit[:hh, :ww] * alpha[:hh, :ww] + out[:hh, :ww] * (1 - alpha[:hh, :ww])

    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
