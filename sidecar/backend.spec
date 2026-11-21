# -*- mode: python ; coding: utf-8 -*-
# PyInstaller one-folder 打包：避免 onefile 的解压路径/启动慢/杀毒误报问题。
# 注意：PyInstaller 不能交叉编译，须在 Windows / Linux 各自构建。
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = []
for mod in ("uvicorn", "fastapi", "pydantic", "PIL", "numpy", "yaml", "psutil"):
    hiddenimports += collect_submodules(mod)

a = Analysis(
    ['backend.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('config', 'config'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'scipy'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name='backend',
)
