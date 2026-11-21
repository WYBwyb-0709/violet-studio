#!/usr/bin/env python3
"""生成 push_files 所需的 files 数组 JSON（base64 内容）。"""
import os
import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 只推送源文件（.gitignore 已排除的产物不在此列，但仍主动过滤）
SKIP_DIRS = {
    "node_modules", ".git", "dist", "build",
    ".venv-build", "__pycache__", "squashfs-root",
}
SKIP_FILES = {".DS_Store"}
INCLUDE_EXT = {
    ".js", ".cjs", ".py", ".html", ".css", ".md", ".yml", ".yaml",
    ".json", ".sh", ".ps1", ".spec", ".txt", ".gitignore", ".svg", ".png",
}
# 特例：无扩展名但需包含
INCLUDE_NAMES = {"LICENSE", "Dockerfile"}


def should_include(rel_path, is_dir):
    parts = rel_path.replace("\\", "/").split("/")
    if any(p in SKIP_DIRS for p in parts[:-1] if is_dir or True):
        return False
    name = parts[-1]
    if is_dir:
        return name not in SKIP_DIRS
    if name in SKIP_FILES:
        return False
    ext = os.path.splitext(name)[1].lower()
    return ext in INCLUDE_EXT or name in INCLUDE_NAMES


def main():
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT).replace("\\", "/")
            if not should_include(rel, False):
                continue
            try:
                with open(full, "rb") as f:
                    content = f.read()
                # 用 git blob hash 逻辑不需要，直接用文本（源码都是 UTF-8）
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    print(f"SKIP binary: {rel}", file=__import__("sys").stderr)
                    continue
                files.append({"path": rel, "content": text})
            except Exception as e:
                print(f"SKIP {rel}: {e}", file=__import__("sys").stderr)
    files.sort(key=lambda x: x["path"])
    out = {"_count": len(files), "files": files}
    with open("/tmp/violet_push_payload.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"files: {len(files)}")
    for x in files:
        print("  ", x["path"], f"({len(x['content'])} chars)")


if __name__ == "__main__":
    main()
