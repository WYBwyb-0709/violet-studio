# Violet Studio

跨平台 **2.5D 分层 AI 生图**桌面软件。一个界面，四种能力：**生成 / 训练 / 模型微调 / 画面调整**。HTML 界面 + Python AI 后端，Windows 与 Linux 双平台安装包，MIT 许可。

> 仓库：https://github.com/WYBwyb-0709/violet-studio

## 核心特性

- **四种模式**
  - 🖼️ **生成**：文生图与分层 RGBA 生成（LayerDiffuse），快/均衡/高质三档。
  - 🎓 **训练**：作品沉淀为数据集，本机 CPU 跑 SD1.5 LoRA 离线训练。
  - 🧩 **模型微调**：LoRA 强度调节、叠加、融合另存。
  - 💡 **画面调整**：调整图层 Z 间距与光照，秒级实时重打光与视差合成——**无需重新生图**。
- **傻瓜式操作**：分步向导 +「一键智能配置」，高级参数可折叠展开。
- **三套紫系主题**：紫+深紫（默认）、浅紫+白、深紫+白，一键切换并持久化。
- **全硬件自适应**：启动探测 CPU 指令集（AVX2 / AVX512 / AMX / ARM NEON）、核心数、内存、GPU，自动选择推理后端与运行参数；可在界面查看并手动覆盖。
- **不爆内存，用时间换结果**：内存预算器预估峰值并自动降级（降分辨率 → 减步数 → 强量化 → 顺序 offload）；VAE tiling 防 decode 阶段 OOM；单并发 + 有界队列 + 任务超时。详见 [内存策略](docs/memory-guide.md)。

## 架构

```
Electron 壳（HTML/CSS/JS + 紫系主题）
   │  contextBridge 白名单 IPC
Main 进程（窗口 / 任务监管 / sidecar 生命周期）
   │  localhost HTTP
Python AI sidecar（PyInstaller one-folder 随包，不进 app.asar）
   ├─ 硬件探测 + 后端注册表 + 内存预算器
   ├─ 生成：OpenVINO GenAI / diffusers / sd.cpp
   ├─ 分层生成：LayerDiffuse 透明 VAE + FG/BG LoRA
   ├─ 2.5D 渲染器：albedo + normal + depth → 重打光 / 视差
   ├─ 数据集导出 + WD14 标注
   └─ CPU LoRA 训练器 + 评测回流
```

场景以结构化 `scene.json` 表示（每层含 RGBA / albedo / normal / depth / z_depth）。**改间距与光照是快路径**（纯 NumPy 渲染，秒级，不占模型内存）；**改图层内容才走慢路径**（AI 生成）。详见 [场景数据结构](docs/scene-schema.md)。

## 硬件适配矩阵

| 硬件 | 后端 | 量化 | 说明 |
|---|---|---|---|
| NVIDIA GPU | diffusers + CUDA | fp16 | 最快，xFormers/SDPA |
| Intel iGPU/XPU/NPU | OpenVINO AUTO | int8 | 核显/NPU 加速 |
| x86 CPU（AMX/AVX512-BF16） | OpenVINO GenAI CPU | int8 | 走 AMX/oneDNN |
| x86 CPU（AVX2） | ONNX Runtime / OpenVINO | int8 | 自动启用指令集 |
| ARM / 极低内存 | sd.cpp (ggml GGUF) | Q4/Q8 | 纯 C/C++，NEON，很慢但能跑 |

线程数按**物理核心数**设置（非逻辑线程，避免超线程争缓存）。完整矩阵见 [硬件矩阵](docs/hardware-matrix.md)。

## 现实约束（务必阅读）

- CPU 推理比中端 GPU 慢约 10–50×；步数是最大成本因子，靠蒸馏少步模型 + 量化 + TAESD + offload 缓解。
- **CPU 训练仅现实支持 SD1.5 LoRA、fp32、关闭训练中采样**，小时级；不支持 fp16/bf16 训练、不支持 SDXL/FLUX 训练。
- 分层生成走 diffusers 后端（OpenVINO IR 不支持透明 VAE）；属性估计模型（Depth-Anything / Marigold / BiRefNet）需在首次使用时下载。
- 大模型不随安装包分发，由应用内下载管理器按可用内存拉取；安装包只含程序与小依赖。
- 完整画世界式的自由笔刷/选区/滤镜不在本期范围。

## CPU LoRA 训练实测

训练器已由骨架补全为**真实可运行的 diffusers + peft LoRA 循环**（`sidecar/app/train_lora_cpu.py`）：读取图片与同名 `.txt` caption、真实反向传播、梯度检查点、余弦学习率、保存 peft safetensors。

- **真实数据集**：[Saranga7/pokemon-dreambooth](https://github.com/Saranga7/pokemon-dreambooth) 的 Pikachu 类（8 张图 + 同名 `.txt` caption），由 `tools/fetch_dataset.py` 经 GitHub MCP API 下载。触发词 `violetpika`。
- **沙箱实测**（8GiB RAM / 5 核 CPU，无 GPU）：用 `tools/build_tiny_sd.py` **纯本地构造**的极小 SD（结构同官方 tiny-sd，87 万参数，无需联网下权重），跑通「数据 → caption → LoRA 注入 → 反向传播 → safetensors → 重新加载出图」全链路。
- **实测结果**：6 步训练 loss 1.192 → 1.146，耗时 6.0s，峰值内存 ~947MB，产出 64 个 LoRA 张量 / 12800 参数；`--validate` 生成 32×32 验证图（std 42.5，非纯色）。
- **目标机**（123GiB / 16c32t）：用完整 SD1.5 权重按下方命令跑真实 LoRA，小时级。

```bash
# 一键准备：下载数据集 + 本地构造 tiny SD（均无需大模型下载）
python tools/fetch_dataset.py
python tools/build_tiny_sd.py

# 沙箱链路验证（tiny SD，秒级）
python sidecar/app/train_lora_cpu.py \
  --model models/tiny-sd-local --data_dir data/pikachu_lora \
  --resolution 64 --max_train_steps 6 --rank 4 --network_alpha 4 \
  --num_cpu_threads 4 --output_dir models/lora/pika_test --validate

# 目标机真实训练（本地 SD1.5 权重）
python sidecar/app/train_lora_cpu.py \
  --model /path/to/sd15 --data_dir /path/to/dataset \
  --resolution 512 --max_train_steps 1500 --rank 16 \
  --gradient_accumulation_steps 4 --num_cpu_threads 16 \
  --output_dir models/lora/my_style
```

CPU 上强制 fp32（`--mixed_precision no`），不支持 fp16/bf16 训练；256GB 以下内存建议 `--cache_latents` 或降分辨率。真实训练元数据见 `models/lora/*/training_meta.json`。

> 说明：沙箱用本地构造的 tiny SD 是为在 8GiB/无 GPU 环境验证训练**代码链路正确性**并产出**真实 LoRA 权重**；其权重随机初始化，出图质量不代表模型效果。目标机加载真实 SD1.5 权重即得可用 LoRA。

## 安装

从 Releases 下载对应安装包：

- **Windows**：`Violet-Studio-*-win-x64.exe`（NSIS 安装向导）
- **Linux**：`Violet-Studio-*-linux-x64.AppImage`（免安装）或 `.deb`

首次启动后，点「分析硬件」→「应用智能配置」，即可开始。模型在首次生成时按需下载。

## 从源码构建

PyInstaller **不能交叉编译**：Windows 与 Linux 安装包须分别在对应平台构建。

```bash
# Linux（产出 AppImage + deb）
bash build/build-linux.sh

# Windows（PowerShell，产出 NSIS exe）
.\build\build-win.ps1
```

或用 GitHub Actions：推送 `v*` 标签触发 `.github/workflows/build.yml`，matrix 分别在 `windows-latest` / `ubuntu-latest` 构建并发布到 Release。

开发模式（不打包后端，UI 用 mock 数据预览）：

```bash
pnpm install
pnpm run dev
```

## 项目结构

```
main/                Electron 主进程 + preload
src/                 HTML/CSS/JS 前端（三主题 + 四模式 + 向导）
sidecar/             Python AI 后端（FastAPI + 各管线 + 配置 + PyInstaller spec）
build/               双平台构建脚本
.github/workflows/   双平台 CI
docs/                架构 / 场景 / 硬件 / 内存文档
```

## 许可

MIT — 见 [LICENSE](LICENSE)。第三方模型与依赖各自遵守其许可（见文档）。
