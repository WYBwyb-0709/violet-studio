/**
 * Electron 主进程
 * 职责：窗口生命周期、启动/监管 Python sidecar、IPC 转发、日志、更新检查占位。
 * 安全：contextIsolation=true、sandbox=true、nodeIntegration=false；preload 白名单 API。
 */
const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const http = require('node:http');

const DEV = process.argv.includes('--dev');
const SIDECAR_PORT = 8765;
let mainWindow = null;
let sidecar = null;
let sidecarReady = false;

function sidecarPath() {
  // 生产环境：extraResources 解压到 process.resourcesPath/backend
  const base = app.isPackaged ? process.resourcesPath : path.join(__dirname, '..', 'sidecar', 'dist', 'backend');
  const exe = process.platform === 'win32' ? 'backend.exe' : 'backend';
  return path.join(base, exe);
}

function startSidecar() {
  const exe = sidecarPath();
  if (!fs.existsSync(exe)) {
    dialog.showErrorBox(
      '后端缺失',
      `找不到 AI 后端程序：\n${exe}\n\n请先运行构建脚本生成 sidecar，或使用开发模式。`
    );
    return;
  }
  sidecar = spawn(exe, [`--port=${SIDECAR_PORT}`], {
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  sidecar.stdout.on('data', (d) => log('[sidecar]', d.toString()));
  sidecar.stderr.on('data', (d) => log('[sidecar:err]', d.toString()));
  sidecar.on('exit', (code) => {
    log('[sidecar] exited with code', code);
    sidecarReady = false;
  });
  pollSidecar();
}

function pollSidecar() {
  const tryOnce = () => {
    const req = http.get({ host: '127.0.0.1', port: SIDECAR_PORT, path: '/api/health', timeout: 1000 }, (res) => {
      res.resume();
      if (res.statusCode === 200) {
        sidecarReady = true;
        mainWindow?.webContents.send('sidecar-ready');
      }
    });
    req.on('error', () => setTimeout(tryOnce, 500));
    req.on('timeout', () => { req.destroy(); setTimeout(tryOnce, 500); });
  };
  tryOnce();
}

function log(...args) {
  const line = args.join(' ') + '\n';
  process.stdout.write(line);
  const logFile = path.join(app.getPath('userData'), 'violet.log');
  try { fs.appendFileSync(logFile, `[${new Date().toISOString()}] ${line}`); } catch (_) {}
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1240,
    height: 820,
    minWidth: 900,
    minHeight: 600,
    title: 'Violet Studio',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(__dirname, '..', 'src', 'index.html'));
  if (DEV) mainWindow.webContents.openDevTools();
}

app.whenReady().then(() => {
  createWindow();
  if (!DEV) startSidecar(); else sidecarReady = true;
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (sidecar && !sidecar.killed) sidecar.kill();
    app.quit();
  }
});

// 白名单 IPC：所有请求转发到本地 sidecar（避免 renderer 直接访问网络/文件）
function apiRequest(method, endpoint, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const req = http.request({
      host: '127.0.0.1', port: SIDECAR_PORT,
      path: endpoint, method,
      headers: data ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } : {},
    }, (res) => {
      let raw = '';
      res.on('data', (c) => (raw += c));
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(raw) }); }
        catch (_) { resolve({ status: res.statusCode, body: raw }); }
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

ipcMain.handle('api:get', (_e, endpoint) => apiRequest('GET', endpoint));
ipcMain.handle('api:post', (_e, endpoint, body) => apiRequest('POST', endpoint, body));
ipcMain.handle('app:status', () => ({ ready: sidecarReady, platform: process.platform }));
ipcMain.handle('app:pickFolder', async () => {
  const r = await dialog.showOpenDialog(mainWindow, { properties: ['openDirectory'] });
  return r.canceled ? null : r.filePaths[0];
});
