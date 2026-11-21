#!/usr/bin/env python3
"""一次性推送全部源码到 GitHub 仓库（使用 MCP 网关）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gh_mcp import call, unwrap

OWNER = "WYBwyb-0709"
REPO = "violet-studio"
BRANCH = "main"


def main():
    payload_path = "/tmp/violet_push_payload.json"
    if not os.path.exists(payload_path):
        print("缺少 payload，请先运行 make_payload.py", file=sys.stderr)
        sys.exit(1)
    with open(payload_path, encoding="utf-8") as f:
        data = json.load(f)
    files = data["files"]
    print(f"推送 {len(files)} 个文件到 {OWNER}/{REPO} ...", flush=True)

    result = call("push_files", {
        "owner": OWNER,
        "repo": REPO,
        "branch": BRANCH,
        "message": "feat: Violet Studio - 跨平台 2.5D 分层 AI 生图桌面软件 (MIT)",
        "files": files,
    }, req_id=100)
    out = unwrap(result)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
