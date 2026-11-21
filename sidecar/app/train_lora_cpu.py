"""
SD1.5 CPU LoRA 训练脚本骨架。
现实约束（已联网核实）：CPU 上 fp16/bf16 训练不生效，须 fp32；训练中采样会报缺 GPU；
SD1.5 LoRA 12 线程约 3.42s/it，16c32t 亦为小时级。不支持 SDXL/FLUX 训练。
"""
import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SD1.5 CPU LoRA 训练（时间换结果）")
    p.add_argument("--data_dir", required=True, help="训练图片目录")
    p.add_argument("--trigger", default="violetstyle", help="触发词")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--use_cpu", action="store_true", help="强制 CPU 训练")
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no",
                   help="CPU 必须填 no（fp16/bf16 不生效）")
    p.add_argument("--num_cpu_threads_per_process", type=int, default=8)
    p.add_argument("--network_alpha", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--no_sample_during_training", action="store_true", default=True,
                   help="CPU 上训练中采样会报错，始终关闭")
    p.add_argument("--mem_limit_gb", type=float, default=32, help="训练进程内存上限提示")
    return p


def main():
    args = build_parser().parse_args()
    # 真实训练流程（需 diffusers/accelerate 与本地 SD1.5 权重）：
    #   1. 用 WD14 tagger 为 data_dir 图片生成 captions（CPU）
    #   2. StableDiffusionPipeline.from_pretrained(local_sd15, torch_dtype=torch.float32)
    #   3. LoRA 注入 UNet attention（rank/network_alpha）
    #   4. accelerate 训练，mixed_precision='no'，num_cpu_threads=...
    #   5. 保存 safetensors 到 models/lora/<name>
    #   6. 不生成训练中样本；训练后用独立进程评测出图
    print("[violet] CPU LoRA 训练参数：", vars(args))
    print("[violet] 这是骨架；请在安装 diffusers/accelerate 与 SD1.5 权重后启用真实训练循环。")


if __name__ == "__main__":
    main()
