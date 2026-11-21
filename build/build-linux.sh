#!/usr/bin/env bash
# Linux 构建：PyInstaller one-folder sidecar + electron-builder AppImage/deb
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== [1/4] Python venv & deps =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r sidecar/requirements.txt pyinstaller
echo "== [2/4] PyInstaller sidecar =="
pyinstaller sidecar/backend.spec --distpath sidecar/dist --workpath sidecar/build --noconfirm
echo "== [3/4] npm deps =="
pnpm install
echo "== [4/4] electron-builder (AppImage + deb) =="
pnpm run build:linux
echo "== 完成：产物在 dist/ =="
ls -lh dist
