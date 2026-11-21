# 硬件适配矩阵

启动时 `sidecar/app/hardware.py` 探测硬件，`registry.py` 据此选择后端与运行参数。所有探测结果在 UI「分析硬件」可见，高级模式可手动覆盖。

## 指令集检测

| 平台 | 检测方式 |
|---|---|
| Linux | 解析 `/proc/cpuinfo` 的 `flags`（avx2 / avx512f / avx512_bf16 / amx_bf16 / amx_int8 / neon / sve） |
| Windows | 读 CPU 名称推断（保守：能识别 AVX2，对 Ice Lake / Granite Rapids / Zen4 等推断 AVX512/AMX） |

## 后端优先级

| 条件 | 后端 | 量化 | 说明 |
|---|---|---|---|
| NVIDIA (cuda) | diffusers + CUDA | fp16 | 最快；xFormers / SDPA |
| Intel iGPU / XPU / NPU | OpenVINO AUTO | int8 | 核显/NPU，AUTO 设备选择 |
| x86 + (AMX 或 AVX512-BF16) | OpenVINO GenAI CPU | int8 | oneDNN / AMX 矩阵单元 |
| x86 + AVX2 | ONNX Runtime / OpenVINO | int8 | 自动启用向量指令 |
| ARM / 无向量指令 | sd.cpp (ggml GGUF) | Q4/Q8 | 纯 C/C++，NEON，兜底 |

## 线程策略

- `intra_op_num_threads` = **物理核心数**（非逻辑线程，避免超线程争 L2 缓存）。
- `inter_op_num_threads` = 1（防多模型线程池竞争）。
- P+E 混合架构（如 Core Ultra / i5-14600KF）：更优为 intra_op = P 核数。
- CPU 训练：`num_cpu_threads_per_process` 取物理核心数一半。

## 时间换结果的档位

| 硬件档 | 内存 | 分辨率 | 后端 | 预期（512 图） |
|---|---|---|---|---|
| 极低 | <4GB | 384 | sd.cpp Q4 | 分钟级 |
| 低 | <8GB | 512 | ONNX int8 + sequential offload | 数十秒~分钟 |
| 中 | <16GB | 768 | OpenVINO int8 + model offload | 十几秒 |
| 高 | ≥16GB | 1024 | OpenVINO int8 | 数秒~十几秒 |

> 数字为量级估算。参考：FastSD CPU 项目在 i7-12700 上，SDXS 1 步 + TAESD 约 0.82s，SDXL-Turbo 2.5s，SDXL-Lightning 768 2 步约 10s；CPU 比中端 GPU 慢 10–50×。

## 已验证环境

本次开发沙箱（非目标机）：AMD EPYC 9K65 / 5 核 / 8.2GB / 无 NVIDIA GPU / 支持 AVX2+AVX512(含 BF16,VNNI) / `/dev/dri` 存在。探测与后端选择逻辑已在此环境跑通。
