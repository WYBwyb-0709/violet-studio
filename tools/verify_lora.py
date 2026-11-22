#!/usr/bin/env python3
"""验证真实 LoRA：同一 prompt 下对比「底模原样」与「注入 LoRA」的生成差异，实时打印进度。"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import PeftModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "models", "tiny-sd")
LORA = os.path.join(ROOT, "models", "lora", "pika_real")
OUT = os.path.join(ROOT, "models", "lora", "pika_real")
PROMPT = "a photo of violetpika, a pikachu character"
STEPS = 20


def gen(pipe, tag):
    print(f"\n[gen:{tag}] 开始去噪 {STEPS} 步", flush=True)
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        img = pipe(prompt=PROMPT, num_inference_steps=STEPS, width=32, height=32,
                   guidance_scale=1.0).images[0]
    path = os.path.join(OUT, f"compare_{tag}.png")
    img.save(path)
    a = np.array(img.convert("RGB"))
    print(f"[gen:{tag}] 完成 {path} mean={a.mean():.2f} std={a.std():.2f}", flush=True)
    return a


def main():
    # 底模用 DDIM scheduler 更适合推理
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE, torch_dtype=torch.float32, safety_checker=None)
    from diffusers import DDIMScheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cpu")

    # 1) 底模原样
    base_arr = gen(pipe, "base")

    # 2) 注入真实训练得到的 LoRA
    unet = PeftModel.from_pretrained(pipe.unet, LORA)
    unet.eval()
    pipe.unet = unet
    lora_arr = gen(pipe, "lora")

    diff = np.abs(base_arr.astype(float) - lora_arr.astype(float))
    print(f"\n[verify] 底模 vs LoRA 平均像素差异: {diff.mean():.3f} "
          f"最大差异: {diff.max():.0f}", flush=True)
    print(f"[verify] LoRA 是否改变输出: {'是 ✅' if diff.mean() > 0.5 else '差异很小'}", flush=True)


if __name__ == "__main__":
    main()
