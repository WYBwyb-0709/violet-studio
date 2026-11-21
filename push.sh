#!/usr/bin/env bash
# 推送到 GitHub（在已配置 GitHub 认证的环境下运行）
# 用法: GITHUB_REPO_URL=https://github.com/<user>/violet-studio.git bash push.sh
set -euo pipefail
cd "$(dirname "$0")"

REPO_URL="${GITHUB_REPO_URL:-}"
if [ -z "$REPO_URL" ]; then
  echo "请设置 GITHUB_REPO_URL，例如："
  echo "  GITHUB_REPO_URL=https://github.com/yourname/violet-studio.git bash push.sh"
  exit 1
fi

echo "== 初始化 git =="
git init
git config user.name "${GIT_AUTHOR_NAME:-Violet Studio Contributors}"
git config user.email "${GIT_AUTHOR_EMAIL:-violet-studio@users.noreply.github.com}"

echo "== 添加文件 =="
git add .
git status --short

echo "== 提交 =="
git commit -m "feat: Violet Studio - 跨平台 2.5D 分层 AI 生图桌面软件 (MIT)"

echo "== 推送 =="
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git push -u origin main

echo "== 完成 =="
echo "创建仓库时请选择 MIT License；或在 GitHub 上手动添加 LICENSE（本仓库已含）。"
echo "推送 v* 标签可触发双平台安装包构建：git tag v0.1.0 && git push --tags"
