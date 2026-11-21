// 冒烟测试：真实加载页面并验证关键交互（mock 掉主进程 IPC，便于无后端环境运行）
// 用法: xvfb-run -a electron test-ui.cjs --no-sandbox
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const ROOT = path.join(__dirname, '..');

// 注册与主进程同名的 handler，用演示数据回退（验证渲染层逻辑，不依赖真实 AI 后端）
ipcMain.handle('app:status', () => ({ ready: false, platform: process.platform }));
ipcMain.handle('app:pickFolder', () => '/demo/scene.json');
ipcMain.handle('api:get', (_e, endpoint) => {
  if (endpoint === '/api/capabilities') return { status: 200, body: {
    cpu: 'Demo CPU', physical_cores: 16, logical_cores: 32,
    isa: ['avx2', 'avx512f', 'amx_bf16'], gpu: null,
    ram_available_gb: 69, recommended: { backend: 'openvino', tier: 'balanced', resolution: 1024, steps: 20, quant: 'int8', offload: 'model' },
  }};
  if (endpoint === '/api/models') return { status: 200, body: { models: ['sd15-base'] } };
  return { status: 200, body: {} };
});
ipcMain.handle('api:post', () => ({ status: 200, body: { ok: true } }));

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    show: false,
    webPreferences: {
      preload: path.join(ROOT, 'main', 'preload.js'),
      contextIsolation: true, sandbox: true, nodeIntegration: false,
    },
  });
  await win.loadFile(path.join(ROOT, 'src', 'index.html'));
  await new Promise((r) => setTimeout(r, 1000));

  const result = await win.webContents.executeJavaScript(`(() => {
    const out = {};
    out.title = document.title;
    out.tabs = document.querySelectorAll('.tab').length;
    out.panels = document.querySelectorAll('.tab-panel').length;
    out.buttons = document.querySelectorAll('button.btn').length;
    out.defaultTheme = document.documentElement.dataset.theme;
    document.querySelector('[data-tab="adjust"]').click();
    out.adjustVisible = !document.getElementById('panel-adjust').classList.contains('hidden');
    out.generateHidden = document.getElementById('panel-generate').classList.contains('hidden');
    const sel = document.getElementById('themeSelect');
    sel.value = 'light-purple'; sel.dispatchEvent(new Event('change'));
    out.themeSwitched = document.documentElement.dataset.theme === 'light-purple';
    document.getElementById('scenePath').value = '/demo/scene.json';
    document.getElementById('btnPickScene').click();
    // 硬件分析 + 智能配置
    document.querySelector('[data-tab="generate"]').click();
    document.getElementById('btnAnalyze').click();
    return out;
  })()`);
  await new Promise((r) => setTimeout(r, 600));
  const layers = await win.webContents.executeJavaScript(`document.querySelectorAll('.layerZ').length`);
  const after = await win.webContents.executeJavaScript(`(() => ({
    profile: document.getElementById('genProfile').value,
    size: document.getElementById('genSize').value,
    hwBadges: document.getElementById('hwBadges').textContent.trim().length > 0,
  }))()`);

  console.log('UI_TEST_RESULT=' + JSON.stringify({ ...result, ...after, layerSliders: layers }));
  app.quit();
});
