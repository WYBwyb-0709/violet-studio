#!/usr/bin/env python3
"""纯本地构造一个结构真实的极小 Stable Diffusion 模型，用于 CPU 训练链路验证。

不依赖网络下载：权重随机初始化，结构与 diffusers 官方 tiny-stable-diffusion-pipe 一致。
tokenizer 使用自建的迷你 CLIP 词表（ASCII + 常用 token），可被 CLIPTokenizer 正常加载。
用途：验证「数据集 -> caption -> LoRA 注入 -> 反向传播 -> safetensors」全链路。
"""
import json
import os

import torch
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

OUT = os.path.join(os.path.dirname(__file__), "..", "models", "tiny-sd-local")
VOCAB_SIZE = 1000


def build_vocab():
    """构造一个可 encode 的迷你 CLIP 词表。"""
    # 用 CLIP 标准 special token，避免 fast tokenizer 保存时 unk 映射丢失
    specials = ["<|startoftext|>", "<|endoftext|>", "<|endoftext|>", "<|endoftext|>"]
    tokens = list(specials)
    # ASCII 可打印字符作为单字符 token
    for i in range(32, 127):
        tokens.append(chr(i))
    # 一些常见英文词与触发词
    extra = [
        "a", "photo", "of", "the", "pikachu", "violetpika", "character",
        "yellow", "electric", "mouse", "pokemon", "in", "on", "with",
        "hello", "world", "cat", "dog", "style", "art", "light", "dark",
    ]
    tokens.extend(extra)
    # 补齐到 VOCAB_SIZE
    i = 0
    while len(tokens) < VOCAB_SIZE:
        tokens.append(f"<extra_{i}>")
        i += 1
    vocab = {t: idx for idx, t in enumerate(tokens)}
    # merges.txt：CLIP BPE 至少要有内容，这里放占位（无合并规则， tokenizer 会按字符切）
    merges = ["#version: 0.2", "a b</w>"]
    return vocab, merges


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) tokenizer
    vocab, merges = build_vocab()
    tok_dir = os.path.join(OUT, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    with open(os.path.join(tok_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(tok_dir, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(merges) + "\n")
    with open(os.path.join(tok_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump({
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
        }, f)
    tokenizer = CLIPTokenizer(
        vocab_file=os.path.join(tok_dir, "vocab.json"),
        merges_file=os.path.join(tok_dir, "merges.txt"),
        model_max_length=32,
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(tok_dir)
    print(f"[build] tokenizer: vocab_size={len(vocab)}", flush=True)

    # 2) text encoder（小隐藏层）
    hidden = 64
    text_cfg = CLIPTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden,
        intermediate_size=hidden * 4,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
        pad_token_id=vocab["<|endoftext|>"],
    )
    text_encoder = CLIPTextModel(text_cfg)
    text_encoder.eval()
    text_encoder.save_pretrained(os.path.join(OUT, "text_encoder"))
    print(f"[build] text_encoder: hidden={hidden}", flush=True)

    # 3) VAE（极小）
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(16,),
        layers_per_block=1,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=4,
        sample_size=32,
    )
    vae.eval()
    vae.save_pretrained(os.path.join(OUT, "vae"))
    print("[build] vae: channels=(16,)", flush=True)

    # 4) UNet（极小，结构与官方 tiny 一致）
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=hidden,
        attention_head_dim=8,
        norm_num_groups=8,
    )
    unet.eval()
    unet.save_pretrained(os.path.join(OUT, "unet"))
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"[build] unet: params={n_params:,}", flush=True)

    # 5) scheduler
    sched = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )
    sched.save_pretrained(os.path.join(OUT, "scheduler"))

    # 6) model_index.json（StableDiffusionPipeline 加载所需）
    model_index = {
        "_class_name": "StableDiffusionPipeline",
        "_diffusers_version": "0.43.0",
        "feature_extractor": [None, None],
        "image_encoder": [None, None],
        "safety_checker": [None, None],
        "scheduler": ["diffusers", "DPMSolverMultistepScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    with open(os.path.join(OUT, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2)

    print(f"[build] 完成：{OUT}", flush=True)


if __name__ == "__main__":
    main()
