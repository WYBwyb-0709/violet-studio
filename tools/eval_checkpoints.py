#!/usr/bin/env python3
"""严谨评估「时间换结果」：修复 peft/diffusers key 兼容问题后，
固定种子+噪声对比各 checkpoint 的去噪 loss，验证 loss 是否随训练下降。
"""
import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "models", "tiny-sd")
DATA = os.path.join(ROOT, "data", "pikachu_lora")
OUT = os.path.join(ROOT, "models", "lora")
CKPTS = [
    ("base", None),
    ("pika_real_40", os.path.join(OUT, "pika_real", "adapter_model.safetensors")),
    ("ckpt_100", os.path.join(OUT, "pika_long", "checkpoint-100", "adapter_model.safetensors")),
    ("ckpt_200", os.path.join(OUT, "pika_long", "checkpoint-200", "adapter_model.safetensors")),
    ("ckpt_300", os.path.join(OUT, "pika_long", "checkpoint-300", "adapter_model.safetensors")),
    ("final_400", os.path.join(OUT, "pika_long", "adapter_model.safetensors")),
]
SEED = 1234
T_STEP = 500


def clean_key(k):
    """修复双重前缀 + 补 .default，映射到模型 named_parameters 的 key。"""
    kk = k.replace("base_model.model.base_model.model.", "base_model.model.")
    kk = kk.replace(".lora_A.weight", ".lora_A.default.weight")
    kk = kk.replace(".lora_B.weight", ".lora_B.default.weight")
    return kk


def build_batch(pipe, proc):
    torch.manual_seed(SEED)
    imgs, caps = [], []
    for fn in sorted(os.listdir(DATA)):
        if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        im = Image.open(os.path.join(DATA, fn)).convert("RGB")
        arr = proc.preprocess(image=np.array(im), height=32, width=32)
        px = arr.float() if isinstance(arr, torch.Tensor) else torch.from_numpy(arr).float()
        if px.dim() == 4:
            px = px.squeeze(0)
        imgs.append(px)
        cf = os.path.splitext(fn)[0] + ".txt"
        caps.append(open(os.path.join(DATA, cf), encoding="utf-8").read().strip() if os.path.exists(os.path.join(DATA, cf)) else "violetpika")
    x = torch.stack(imgs)
    with torch.no_grad():
        lat = pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor
    enc = pipe.tokenizer(caps, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hs = pipe.text_encoder(**enc)[0]
    noise = torch.randn_like(lat)
    return lat, hs, noise


def main():
    torch.set_num_threads(4)
    proc = VaeImageProcessor(do_resize=True, do_convert_rgb=True, do_normalize=True)
    pipe = StableDiffusionPipeline.from_pretrained(BASE, dtype=torch.float32, safety_checker=None)
    sched = DDPMScheduler.from_config(pipe.scheduler.config)
    lat, hs, noise = build_batch(pipe, proc)
    t = torch.full((lat.shape[0],), T_STEP, dtype=torch.long)
    noisy = sched.add_noise(lat, noise, t)

    print("\n[eval] 固定种子/噪声/时间步=500，训练集去噪 MSE（越低越拟合）:", flush=True)
    results = []
    for tag, path in CKPTS:
        # 每次全新构造 base + 注入，避免对象污染
        p2 = StableDiffusionPipeline.from_pretrained(BASE, dtype=torch.float32, safety_checker=None)
        cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0"], bias="none")
        m = get_peft_model(p2.unet, cfg)
        m.eval()
        if path and os.path.exists(path):
            sd = load_file(path)
            nmap = dict(m.named_parameters())
            loaded = 0
            for k, v in sd.items():
                ck = clean_key(k)
                if ck in nmap:
                    nmap[ck].data.copy_(v.float())
                    loaded += 1
        else:
            loaded = 0
        with torch.no_grad():
            pred = m(noisy, t, hs).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float()).item()
        results.append((tag, loss))
        print(f"  {tag:<14} loss={loss:.6f}  (loaded {loaded} tensors)", flush=True)

    base = results[0][1]
    print("\n[eval] 相对底模改善:", flush=True)
    for tag, loss in results:
        print(f"  {tag:<14} {(base-loss)/base*100:+6.3f}%", flush=True)


if __name__ == "__main__":
    main()
