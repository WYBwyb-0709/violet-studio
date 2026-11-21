# MultiLayerScene 场景数据结构

作品以「一个 `scene.json` + 每层图片文件」的结构存储，是"画世界"式分层编辑的核心数据契约。

## 目录结构

```
my-scene/
  scene.json
  layers/
    bg.png          # RGBA 可见图
    bg_albedo.png   # 去光照反照率（重打光前提）
    bg_normal.png   # 法线（表面起伏）
    bg_depth.png    # 深度（视差 + AO）
    mid.png / mid_albedo.png / ...
```

## scene.json 示例

```json
{
  "version": "1.0",
  "canvas": { "width": 1024, "height": 1024 },
  "camera": { "type": "parallax", "fov": 45 },
  "lighting": {
    "key": { "azimuth": 45, "elevation": 45, "intensity": 1.0 },
    "fill": { "intensity": 0.4 },
    "ambient": 0.6,
    "ao": 0.8
  },
  "layers": [
    {
      "id": 0,
      "name": "背景",
      "z_depth": 0.0,
      "opacity": 1.0,
      "blend_mode": "normal",
      "parallax_factor": 0.0,
      "visible": true,
      "locked": false,
      "rgba": "layers/bg.png",
      "albedo": "layers/bg_albedo.png",
      "normal": "layers/bg_normal.png",
      "depth": "layers/bg_depth.png"
    },
    {
      "id": 1,
      "name": "主体",
      "z_depth": 30.0,
      "opacity": 1.0,
      "visible": true,
      "rgba": "layers/mid.png",
      "albedo": "layers/mid_albedo.png",
      "normal": "layers/mid_normal.png",
      "depth": "layers/mid_depth.png"
    }
  ]
}
```

## 字段说明

| 字段 | 含义 | 编辑时影响 |
|---|---|---|
| `z_depth` | 层的前后深度，越大越靠近观察者 | 改间距 → 视差变化（快路径） |
| `lighting.*` | 主光方位/高度/强度、补光、环境光、AO | 改光照 → 逐像素重打光（快路径） |
| `rgba` | 可见透明图 | 合成用 |
| `albedo` | 去光照反照率 | **重打光的前提**；无此图只能整体调亮度 |
| `normal` | 法线图 | 表面起伏与光照方向响应 |
| `depth` | 深度图 | 视差 + 环境光遮蔽（AO） |
| `opacity` | 不透明度 | 合成混合 |
| `visible` | 是否可见 | 合成时跳过 |

## 快路径 vs 慢路径

- **快路径**（改 `z_depth`、`lighting.*`、`opacity`）：`renderer.render()` 用 albedo+normal+depth 纯 NumPy 重算，秒级，不调用 AI、不占模型内存。
- **慢路径**（改图层内容）：LayerDiffuse 生成 RGBA + 属性估计模型生成 albedo/normal/depth，分钟级。

## 图层来源

- AI 生成：LayerDiffuse（SD1.5 透明 VAE + FG/BG LoRA）。
- 属性估计：Depth-Anything-V2-Base（深度，Apache-2.0）、Marigold-normals / Marigold-IID（法线/反照率）、BiRefNet（抠图，MIT）。
- **注意**：Depth-Anything **Large** 为 CC-BY-NC，商用不可用；只用 Base。
