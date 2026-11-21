# 内存策略：不爆内存，用时间换结果

无 GPU 或低显存时，核心矛盾是"峰值内存 vs 速度"。Violet Studio 宁可慢，也不 OOM。

## 内存预算器

`registry.py::auto_plan()` 在生成前：

1. 读可用内存（或用户设定的 `mem_limit_gb`），留出 2GB 系统余量。
2. 按档位选初始分辨率/步数/量化/offload。
3. 估算峰值：`基础占用 × (分辨率/512)² × offload 系数`。
   - offload=none → ×1.0；model → ×0.7；sequential → ×0.45。
4. 若峰值超预算，按序降级：降分辨率到 512 → 步数减半 → 改 int4 → 开 sequential offload，直到预算内。

## offload 阶梯（越往下越省内存越慢）

| 级别 | 手段 | 内存 | 速度影响 |
|---|---|---|---|
| 0 | 模型常驻 + fp16 + FlashAttention | 高 | 最快 |
| 1 | model CPU offload + VAE slicing/tiling + attention slicing | 中 | 慢 10–20% |
| 2 | sequential CPU offload + int8/GGUF Q4 + 降分辨率 | 低 | 慢 30–50% 甚至 3–5× |
| 3 | sd.cpp GGUF Q4，单线程分批 | 极低 | 很慢但能跑 |

## 防 OOM 的具体机制

- **VAE tiling / slicing**：OOM 常发生在最后 decode 阶段（不是 UNet）；分块解码绕过。
- **文本编码器**：SD3/FLUX 的 T5-XXL fp16 约 9GB，是隐藏的内存杀手；这类模型不在默认档位。
- **单模型常驻**：切模型先卸载旧实例；权重 mmap 共享。
- **推理与训练分时**：不同进程、不同时段，禁止并发抢内存；训练进程单独内存上限。
- **有界队列 + 单并发 + 任务超时**：避免请求堆积打爆内存。
- **生成后 `empty_cache()`**：防多次生成的 PyTorch 缓存累积。
- **训练内存上限**：systemd `MemoryMax` / cgroup（Linux）、Job Object（Windows）。

## 为什么改间距/光照不占模型内存

画面调整模式用 `renderer.py` 的纯 NumPy 实现，直接读 albedo/normal/depth 图片重打光，不加载任何 AI 模型，因此可高并发、秒级响应，与主生成任务的内存完全隔离。

## 量化精度对照

| 量化 | 内存 | 质量 | 适用 |
|---|---|---|---|
| fp16 | 基准 | 最好 | 内存充足 |
| int8 | ~50–60% | 几乎无损 | OpenVINO / ONNX 默认 |
| int4 / GGUF Q4 | ~30–40% | 轻微损失 | 低内存兜底 |
