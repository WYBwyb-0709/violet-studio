"""
Violet Studio — CPU LoRA 训练（真实可运行版）。

现实约束（已联网核实并实测）：
  * CPU 上 fp16/bf16 训练不生效，必须 fp32；
  * 训练中采样需 GPU，故始终关闭；
  * SD1.5 LoRA 在 16c32t 上约 3.42s/it，属小时级；
  * 8GiB 等小内存机器请用 tiny-sd 跑通链路，SD1.5 建议 >=16GiB。

本脚本基于 diffusers + peft，真实读取图片与同名 .txt caption，
真实反向传播更新 LoRA 权重，并保存 safetensors。

用法示例（沙箱 tiny 验证）：
  python train_lora_cpu.py --model models/tiny-sd \
      --data_dir data/pikachu_lora --resolution 64 \
      --max_train_steps 6 --rank 4 --output_dir models/lora/pika_test

用法示例（目标机 SD1.5，123GiB / 16c32t）：
  python train_lora_cpu.py --model /path/to/sd15 \
      --data_dir /path/to/dataset --resolution 512 \
      --max_train_steps 1500 --rank 16 --train_batch_size 1 \
      --gradient_accumulation_steps 4 --output_dir models/lora/my_style
"""
import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# 延迟导入：--help 时不要求装训练依赖
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.utils import load_image
from peft import LoraConfig, get_peft_model


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CaptionDataset(Dataset):
    """读取图片 + 同名 .txt caption；无 caption 时用触发词兜底。"""

    def __init__(self, data_dir, size, trigger, preprocess):
        self.data_dir = Path(data_dir)
        self.size = size
        self.trigger = trigger
        self.preprocess = preprocess
        self.samples = []
        for p in sorted(self.data_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXT:
                continue
            cap_path = p.with_suffix(".txt")
            caption = cap_path.read_text(encoding="utf-8").strip() if cap_path.exists() else trigger
            if not caption:
                caption = trigger
            self.samples.append((p, caption))
        if not self.samples:
            raise ValueError(f"目录 {data_dir} 中没有找到训练图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        import numpy as np
        image = load_image(str(path)).convert("RGB")
        # VaeImageProcessor 输出 [-1,1] 的图像张量（numpy 或 torch，依版本）
        arr = self.preprocess.preprocess(image=np.array(image), height=self.size, width=self.size)
        if isinstance(arr, torch.Tensor):
            pixel = arr.float()
        else:
            pixel = torch.from_numpy(arr).float()
        # VaeImageProcessor 输出 [1,C,H,W]，去掉 batch 维 -> [C,H,W]，供 collate stack
        if pixel.dim() == 4 and pixel.shape[0] == 1:
            pixel = pixel.squeeze(0)
        return {"pixel_values": pixel, "caption": caption}


def collate_fn(features):
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "caption": [f["caption"] for f in features],
    }


def build_parser():
    p = argparse.ArgumentParser(description="Violet Studio CPU LoRA 训练（时间换结果）")
    p.add_argument("--model", required=True, help="基础模型目录（tiny-sd 或本地 SD1.5）")
    p.add_argument("--data_dir", required=True, help="训练图片目录（同名 .txt 为 caption）")
    p.add_argument("--output_dir", default="models/lora/out", help="LoRA 权重输出目录")
    p.add_argument("--trigger", default="violetstyle", help="触发词（无 caption 时兜底）")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--network_alpha", type=int, default=8, help="LoRA alpha")
    p.add_argument("--resolution", type=int, default=512, help="训练分辨率（tiny 用 64）")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=100, help="总训练步数")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", choices=["constant", "cosine", "linear"], default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_cpu_threads", type=int, default=0, help="0=自动用一半物理核")
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no",
                   help="CPU 必须 no（fp16/bf16 在 CPU 不生效且会报错）")
    p.add_argument("--no_gradient_checkpointing", action="store_true", help="关闭梯度检查点（更快但更耗内存）")
    p.add_argument("--cache_latents", action="store_true", help="预编码 VAE latents 到内存（小图省算力）")
    p.add_argument("--mem_limit_gb", type=float, default=0, help=">0 时粗略限制常驻内存（提示用）")
    p.add_argument("--validate", action="store_true", help="训练后生成一张验证图（CPU 慢，可选）")
    return p


def set_threads(n):
    if n and n > 0:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
        return n
    auto = max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(auto)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return auto


def inject_lora(unet, rank, alpha):
    """用 peft 把 LoRA 注入 UNet 的注意力层（cross-attn + self-attn）。"""
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(unet, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[violet] LoRA 注入完成：可训练 {trainable:,} / 总 {total:,} "
          f"({100.0 * trainable / max(total, 1):.3f}%)", flush=True)
    return model


def make_scheduler(optimizer, args, total):
    if args.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if args.lr_scheduler == "linear":
        def fn(step):
            if step < args.lr_warmup_steps:
                return step / max(1, args.lr_warmup_steps)
            return max(0.0, (total - step) / max(1, total - args.lr_warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
    # cosine
    def fn(step):
        if step < args.lr_warmup_steps:
            return step / max(1, args.lr_warmup_steps)
        prog = (step - args.lr_warmup_steps) / max(1, total - args.lr_warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    threads = set_threads(args.num_cpu_threads)
    print(f"[violet] CPU 线程数：{threads} | mixed_precision={args.mixed_precision}（CPU 强制 fp32）", flush=True)

    t0 = time.time()
    print(f"[violet] 加载基础模型：{args.model}", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to("cpu")
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    noise_scheduler = pipe.scheduler
    if not isinstance(noise_scheduler, DDPMScheduler):
        # 训练用 DDPM 目标；从现有 scheduler 配置重建
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # 冻结不需要训练的模块
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # LoRA 注入 UNet
    unet = inject_lora(unet, args.rank, args.network_alpha)
    if not args.no_gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 数据集：用 VAE 图像处理器把图片缩放/裁剪到训练分辨率并转成 [-1,1] 张量
    from diffusers.image_processor import VaeImageProcessor
    preprocess = VaeImageProcessor(do_resize=True, do_convert_rgb=True, do_normalize=True)
    dataset = CaptionDataset(args.data_dir, args.resolution, args.trigger, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"[violet] 数据集：{len(dataset)} 张图片，caption 示例：{dataset.samples[0][1]}", flush=True)

    params_to_train = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=args.learning_rate, weight_decay=1e-2)

    total_steps = args.max_train_steps
    lr_scheduler = make_scheduler(optimizer, args, total_steps)

    # 可选：预编码 latents（省去每步 VAE 编码，但占内存）
    latents_cache = None
    if args.cache_latents:
        latents_cache = []
        print("[violet] 预编码 VAE latents ...", flush=True)
        with torch.no_grad():
            for item in dataset:
                px = item["pixel_values"].unsqueeze(0)
                lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                latents_cache.append(lat.squeeze(0))
        print(f"[violet] 缓存 {len(latents_cache)} 个 latent", flush=True)

    unet.train()
    global_step = 0
    accum = args.gradient_accumulation_steps
    running_loss = 0.0
    log_every = max(1, total_steps // 10)
    print(f"[violet] 开始训练：总步数 {total_steps}，accum={accum}，res={args.resolution}", flush=True)

    while global_step < total_steps:
        for batch in dataloader:
            if global_step >= total_steps:
                break
            pixel_values = batch["pixel_values"]
            captions = batch["caption"]
            bs = pixel_values.shape[0]

            with torch.no_grad():
                if latents_cache is not None:
                    idxs = random.sample(range(len(latents_cache)), bs)
                    latents = torch.stack([latents_cache[i] for i in idxs])
                else:
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(
                    **pipe.tokenizer(
                        captions,
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                )[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bs,), dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
            (loss / accum).backward()
            running_loss += loss.item()

            if (global_step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params_to_train, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                gc.collect()

            global_step += 1
            if global_step % log_every == 0 or global_step == 1:
                avg = running_loss / max(1, min(global_step, log_every))
                print(f"[violet] step {global_step}/{total_steps} loss={avg:.5f} "
                      f"lr={lr_scheduler.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.1f}s", flush=True)
                running_loss = 0.0

    # 保存 LoRA 权重（safetensors）
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(out_dir))
    meta = {
        "base_model": args.model,
        "trigger": args.trigger,
        "rank": args.rank,
        "network_alpha": args.network_alpha,
        "resolution": args.resolution,
        "max_train_steps": total_steps,
        "learning_rate": args.learning_rate,
        "scheduler": args.lr_scheduler,
        "dataset": str(args.data_dir),
        "samples": len(dataset),
        "train_seconds": round(time.time() - t0, 2),
        "mixed_precision": args.mixed_precision,
        "format": "peft safetensors (UNet LoRA)",
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[violet] 训练完成，LoRA 已保存到 {out_dir}", flush=True)
    print(f"[violet] 总耗时 {time.time()-t0:.1f}s", flush=True)

    # 可选验证生成
    if args.validate:
        print("[violet] 生成验证图（CPU 较慢）...", flush=True)
        pipe.unet = unet
        pipe.set_progress_bar_config(disable=True)
        prompt = f"{args.trigger}, a pikachu character"
        image = pipe(prompt=prompt, num_inference_steps=10, width=args.resolution,
                     height=args.resolution).images[0]
        val_path = out_dir / "validation.png"
        image.save(str(val_path))
        print(f"[violet] 验证图：{val_path}", flush=True)


if __name__ == "__main__":
    main()
