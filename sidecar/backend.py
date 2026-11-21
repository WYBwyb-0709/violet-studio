"""
Violet Studio Python AI sidecar 入口（FastAPI）。
Electron 主进程通过 localhost:PORT 与其通信。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 支持 PyInstaller one-folder：把 sidecar 根目录加入 sys.path
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
sys.path.insert(0, str(BASE))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
import uvicorn  # noqa: E402

from app import pipelines, renderer  # noqa: E402
from app.hardware import detect  # noqa: E402
from app.registry import auto_plan  # noqa: E402

app = FastAPI(title="Violet Studio Sidecar", version="0.1.0")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/capabilities")
def capabilities():
    h = detect()
    plan = auto_plan(h)
    d = h.as_dict()
    d["recommended"] = {
        "backend": plan.backend, "tier": plan.tier, "resolution": plan.resolution,
        "steps": plan.steps, "threads": plan.threads, "quant": plan.quant, "offload": plan.offload,
        "reason": plan.reason,
    }
    return d


@app.get("/api/models")
def models():
    d = Path.home() / ".violet-studio" / "models"
    found = []
    if d.exists():
        for p in d.rglob("*"):
            if p.suffix in (".safetensors", ".bin", ".xml"):
                found.append(p.stem)
    return {"models": found or ["sd15-base", "sd15-layerdiffuse"]}


@app.post("/api/generate")
def generate(body: dict):
    tid = pipelines.run_async(pipelines.generate_task, body)
    return {"task_id": tid}


@app.get("/api/task/{tid}")
def task(tid: str):
    with pipelines._lock:
        return JSONResponse(pipelines.TASKS.get(tid, {"state": "not_found"}))


@app.post("/api/task/{tid}/cancel")
def cancel(tid: str):
    pipelines._set(tid, state="cancelled", stage="已取消")
    return {"ok": True}


@app.post("/api/dataset/export")
def dataset_export(body: dict):
    return pipelines.export_dataset(body)


@app.post("/api/train/start")
def train_start(body: dict):
    tid = pipelines.run_async(pipelines.train_task, body)
    return {"task_id": tid}


@app.post("/api/model/load")
def model_load(body: dict):
    return {"ok": True, "note": "模型加载骨架；真实加载需本地权重。"}


@app.post("/api/model/merge")
def model_merge(body: dict):
    return {"ok": True, "note": "LoRA 融合骨架；真实融合需 diffusers load_lora_weights + save。"}


@app.post("/api/render")
def render(body: dict):
    scene = body.get("scene", {})
    if isinstance(scene, str):
        try:
            scene = json.loads(Path(scene).read_text(encoding="utf-8"))
        except Exception:
            return JSONResponse({"detail": "无法读取场景文件"}, status_code=400)
    img = renderer.render(scene, body.get("lighting", {}))
    return {"image": renderer.to_data_uri(img)}


@app.post("/api/export/image")
def export_image(body: dict):
    return {"ok": True, "note": "图片导出骨架"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
