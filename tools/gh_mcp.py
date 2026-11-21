#!/usr/bin/env python3
"""通过 HTTP MCP 网关调用 GitHub 工具（处理 SSE 响应解析）。

用法:
  python3 gh_mcp.py get_me '{}'
  python3 gh_mcp.py create_repository '{"name":"...","description":"..."}'
  python3 gh_mcp.py push_files '{"owner":"...","repo":"...",...}'
"""
import sys
import json
import urllib.request

URL = "http://github.agent-gateway.auth-proxy.local/mcp"
TOKEN = "auto_proxy_token"


def call(method, params, req_id=1):
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": method, "arguments": params},
    }).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {TOKEN}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read().decode("utf-8", "replace")

    # 解析 SSE: 找 data: 行里的 JSON-RPC 响应
    result = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "result" in obj:
                result = obj["result"]
            elif "error" in obj:
                return {"error": obj["error"]}
    return result if result is not None else {"raw": raw[:2000]}


def unwrap(result):
    """GitHub MCP 返回 content[0].text (JSON 字符串)。"""
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return result
    content = result.get("content", [])
    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
    if not texts:
        return result
    text = "\n".join(texts)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


if __name__ == "__main__":
    tool = sys.argv[1]
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    res = call(tool, args)
    out = unwrap(res)
    print(json.dumps(out, ensure_ascii=False, indent=2))
