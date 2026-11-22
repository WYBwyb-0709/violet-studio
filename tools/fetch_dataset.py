#!/usr/bin/env python3
"""从 GitHub 下载 Pikachu 训练图片并生成 LoRA caption。

数据集来源：https://github.com/Saranga7/pokemon-dreambooth (Pikachu 类)
本仓库已自带下载结果 data/pikachu_lora/；此脚本用于重新抓取/扩充。

用法:
  python tools/fetch_dataset.py                 # 下载到默认目录
  GITHUB_TOKEN=xxx python tools/fetch_dataset.py  # 带 token 避免 API 限流

依赖：仅标准库。使用系统 CA 证书，兼容无 certifi 的环境。
"""
import base64
import json
import os
import ssl
import sys
import time
import urllib.request

OWNER = "Saranga7"
REPO = "pokemon-dreambooth"
REMOTE_DIR = "dataset/Pikachu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "pikachu_lora")
TRIGGER = "violetpika"
WANTED = [
    "00000000.jpg", "00000001.png", "00000003.jpg", "00000004.jpg",
    "00000009.png", "00000010.png", "00000011.png", "00000014.png",
]

CA = "/etc/ssl/certs/ca-certificates.crt"
CTX = ssl.create_default_context(cafile=CA) if os.path.exists(CA) else ssl.create_default_context()
TOKEN = os.environ.get("GITHUB_TOKEN", "")


def api_get(path):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "violet-studio",
        "Accept": "application/vnd.github+json",
        **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}),
    })
    with urllib.request.urlopen(req, timeout=40, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))


def download(relname, retries=3):
    for attempt in range(1, retries + 1):
        try:
            meta = api_get(f"{REMOTE_DIR}/{relname}")
            # 大文件 GitHub contents API 返回 content(base64) 或 download_url
            if meta.get("content"):
                return base64.b64decode(meta["content"])
            raw_url = meta.get("download_url")
            if raw_url:
                req = urllib.request.Request(raw_url, headers={"User-Agent": "violet-studio"})
                with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                    return r.read()
        except Exception as e:
            print(f"  retry {attempt}/{retries} {relname}: {str(e)[:80]}", flush=True)
            time.sleep(2 * attempt)
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0
    for name in WANTED:
        data = download(name)
        if not data or len(data) > 2_000_000:
            print(f"  跳过 {name}", flush=True)
            continue
        # 修正扩展名（raw 上 png/jpg 与实际内容可能不符，统一按 MIME 猜）
        ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
        stem = os.path.splitext(name)[0]
        with open(os.path.join(OUT_DIR, f"{stem}{ext}"), "wb") as f:
            f.write(data)
        caption = f"a photo of {TRIGGER}, a pikachu character, yellow electric mouse pokemon"
        with open(os.path.join(OUT_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(caption)
        count += 1
        print(f"  ok {stem}{ext} ({len(data)} bytes)", flush=True)
    print(f"\n完成：{count} 张，目录 {OUT_DIR}", flush=True)
    if count == 0:
        print("提示：GitHub API 匿名限流时设置 GITHUB_TOKEN，或直接使用仓库自带的 data/pikachu_lora/", flush=True)


if __name__ == "__main__":
    main()
