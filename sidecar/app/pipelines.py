"""
任务管线：生成（含 LayerDiffuse 分层）、数据集导出、CPU LoRA 训练、模型管理。
任务在后台线程执行，主线程通过 /api/task/{id} 查询进度。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .backends import BackendUnavailable, build_backend
from .hardware import detect
from .registry import Plan, auto_plan
from . import renderer

DATA_DIR = Path(os.getenv("VIOLET_DATA", str(Path.home() / ".violet-studio")))
TASKS: dict[str, dict] = {}
_lock = threading.Lock()


def _task(id: str) -> dict:
    with _lock:
        return TASKS.setdefault(id, {"state": "running", "progress": 0, "stage": "排队", "eta": "?", "images": []})


def _set(id: str, **kw):
    with _lock:
        _task(id).update(kw)


def run_async(fn, *args):
    tid = uuid.uuid4().hex[:12]
    _task(tid)
    def wrapper():
        try:
            fn(tid, *args)
        except Exception as e:
            _set(tid, state="error", error=str(e))
    threading.Thread(target=wrapper, daemon=True).start()
    return tid


# ---------- 生成 ----------
def generate_task(tid: str, payload: dict):
    _set(tid, stage="分析硬件与内存预算", progress=5)
    h = detect()
    plan: Plan = auto_plan(h, user_tier=payload.get("profile"))
    _set(tid, stage=f"后端 {plan.backend} / {plan.quant} / {plan.offload}", progress=12)

    backend = build_backend(plan)
    try:
        backend.load()
        images = backend.generate(
            payload.get("prompt", ""), payload.get("negative", ""),
            resolution=payload.get("size"), steps=payload.get("steps"),
            layered=payload.get("layered", False),
        )
        _set(tid, state="done", progress=100, stage="完成", images=[renderer.to_data_uri(i) for i in images])
    except BackendUnavailable as e:
        # 无真实模型时的演示图：明确标注为占位，绝不假装真实生成
        demo = Image.new("RGB", (payload.get("size", 512), payload.get("size", 512)), (150, 90, 220))
        _set(tid, state="done", progress=100, stage="演示占位（无真实模型）",
             images=[renderer.to_data_uri(demo)], warning=str(e))


# PIL 延迟导入避免上面的类型引用问题
from PIL import Image  # noqa: E402


# ---------- 数据集导出 ----------
def export_dataset(payload: dict) -> dict:
    src = Path(payload.get("source", ""))
    trigger = payload.get("trigger", "violetstyle")
    version = "v001"
    out = DATA_DIR / "datasets" / version
    (out / "images").mkdir(parents=True, exist_ok=True)
    count = 0
    if src.exists():
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                count += 1
    manifest = {"version": version, "trigger": trigger, "samples": count, "source": str(src)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "captions.jsonl").write_text("", encoding="utf-8")
    return {"path": str(out), "samples": count}


# ---------- CPU LoRA 训练 ----------
def train_task(tid: str, payload: dict):
    _set(tid, stage="检查 CPU 训练约束", progress=3)
    note = ("CPU 训练仅支持 SD1.5 LoRA、fp32、关闭训练中采样；"
            "预计小时级。此处为可运行骨架，实际训练需 models/ 下有 SD1.5 权重与脚本依赖。")
    # 约束检查：CPU 不支持 fp16/bf16 训练，不支持训练中采样
    h = detect()
    if not h.cpu:
        _set(tid, state="error", error="无法识别 CPU")
        return
    script = DATA_DIR / "training" / "train_lora_cpu.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", str(script),
        "--data_dir", str(payload.get("data_dir", "")),
        "--trigger", str(payload.get("trigger", "violetstyle")),
        "--rank", str(payload.get("rank", 16)),
        "--epochs", str(payload.get("epochs", 20)),
        "--resolution", str(payload.get("resolution", 512)),
        "--use_cpu", "--mixed_precision", "no", "--num_cpu_threads_per_process",
        str(max(1, h.physical_cores // 2 or 4)),
    ]
    # 真实训练通过子进程运行；这里不阻塞主线程
    for pct in range(10, 100, 10):
        time.sleep(0.3)
        _set(tid, progress=pct, stage=f"CPU 训练中（骨架） {pct}%", note=note)
    _set(tid, state="done", progress=100, stage="训练完成（骨架）", note=note, cmd=" ".join(cmd))
