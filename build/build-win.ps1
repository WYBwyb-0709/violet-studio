# Windows 构建：PyInstaller one-folder sidecar + electron-builder NSIS
# 在 Windows PowerShell / CI windows-latest 上运行
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
# Force UTF-8 so pip/Python never hit cp1252 decoding errors on Windows
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Write-Host "== [1/4] Python venv & deps =="
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r .\sidecar\requirements.txt pyinstaller
Write-Host "== [2/4] PyInstaller sidecar =="
pyinstaller .\sidecar\backend.spec --distpath .\sidecar\dist --workpath .\sidecar\build --noconfirm
Write-Host "== [3/4] npm deps =="
pnpm install
Write-Host "== [4/4] electron-builder (NSIS exe) =="
pnpm run build:win
Write-Host "== 完成：产物在 dist\ =="
Get-ChildItem .\dist | Select-Object Name,Length
