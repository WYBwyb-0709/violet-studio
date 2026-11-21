/** UI 交互逻辑：主题切换、Tab、硬件分析、生成、训练、微调、画面调整。 */
(function () {
  const $ = (id) => document.getElementById(id);
  let caps = null;
  let currentTask = null;
  const trainWizard = new Wizard(['准备数据', '参数配置', '训练', '评测回流']);

  function toast(msg, ms = 3200) {
    const t = $('toast');
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove('show'), ms);
  }

  // ---- 主题 ----
  const THEME_KEY = 'violet-theme';
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    $('themeSelect').value = t;
    try { localStorage.setItem(THEME_KEY, t); } catch (_) {}
  }
  applyTheme((() => { try { return localStorage.getItem(THEME_KEY) || 'purple'; } catch (_) { return 'purple'; } })());
  $('themeSelect').addEventListener('change', (e) => applyTheme(e.target.value));

  // ---- Tab ----
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((p) => p.classList.add('hidden'));
      tab.classList.add('active');
      $('panel-' + tab.dataset.tab).classList.remove('hidden');
      if (tab.dataset.tab === 'finetune') loadModels();
    });
  });

  // ---- 状态 / 硬件 ----
  async function refreshStatus() {
    if (MOCK) {
      $('statusDot').classList.add('online');
      $('statusText').textContent = 'AI 引擎已就绪（演示模式）';
      return;
    }
    const s = await window.violet.status();
    $('statusDot').classList.toggle('online', !!s.ready);
    $('statusText').textContent = s.ready ? 'AI 引擎已就绪' : 'AI 引擎启动中…';
  }
  refreshStatus();

  function renderBadges(c) {
    const isa = (c.isa || []).slice(0, 4).map((x) => `<span class="badge good">${x}</span>`).join('');
    const tierMap = { fast: '快速', balanced: '均衡', quality: '高质' };
    $('hwBadges').innerHTML =
      `<span class="badge">${c.physical_cores || '?'} 核 / ${c.logical_cores || '?'} 线程</span>` +
      `<span class="badge">${c.ram_available_gb || '?'} GB 可用</span>` +
      isa +
      `<span class="badge warn">建议 ${tierMap[c.recommended?.tier] || '均衡'}</span>`;
  }

  async function analyze() {
    try {
      caps = await apiGet('/api/capabilities');
      renderBadges(caps);
      const r = caps.recommended || {};
      $('smartHint').innerHTML =
        `检测到 <b>${caps.cpu || '未知 CPU'}</b> · ${caps.physical_cores} 核 / ${caps.logical_cores} 线程 · ` +
        `可用内存 ${caps.ram_available_gb} GB · 加速器 ${caps.gpu || '无（纯 CPU）'}<br>` +
        `推荐后端 <b>${r.backend}</b>，档位 <b>${r.tier}</b>，量化 <b>${r.quant}</b>，内存策略 <b>${r.offload}</b>。`;
      toast('硬件分析完成');
    } catch (e) { toast('分析失败：' + e.message); }
  }
  $('btnAnalyze').addEventListener('click', analyze);

  $('btnSmart').addEventListener('click', () => {
    if (!caps) { toast('请先点击“分析硬件”'); return; }
    const r = caps.recommended || {};
    $('genProfile').value = r.tier || 'balanced';
    $('genSize').value = String(r.resolution || 1024);
    $('genSteps').value = r.steps || 20;
    toast('已应用智能配置：' + (r.tier || 'balanced'));
  });

  // ---- 生成 ----
  function progress(elBar, elText, pct, text) {
    elBar.style.width = pct + '%';
    elText.textContent = text;
  }
  function pollTask(id, bar, text, onDone) {
    const timer = setInterval(async () => {
      try {
        const s = await apiGet('/api/task/' + id);
        progress(bar, text, s.progress || 0, `${s.stage || '处理中'} ${s.progress || 0}% · 预计剩余 ${s.eta || '?'}s`);
        if (s.state === 'done') { clearInterval(timer); onDone(s); }
        if (s.state === 'error') { clearInterval(timer); toast('任务失败：' + (s.error || '未知错误')); }
      } catch (_) { /* 轮询容错 */ }
    }, 1200);
    currentTask = { timer, id };
  }

  $('btnGenerate').addEventListener('click', async () => {
    const payload = {
      prompt: $('genPrompt').value.trim(),
      negative: $('genNeg').value.trim(),
      profile: $('genProfile').value,
      size: Number($('genSize').value),
      steps: Number($('genSteps').value),
      layered: $('genLayered').checked,
    };
    if (!payload.prompt) { toast('请输入提示词'); return; }
    $('genProgressWrap').classList.remove('hidden');
    progress($('genProgressBar'), $('genProgressText'), 3, '提交任务…');
    try {
      const r = await apiPost('/api/generate', payload);
      if (MOCK || !r.task_id) { toast('演示模式：已收到生成请求（无真实图像）'); return; }
      pollTask(r.task_id, $('genProgressBar'), $('genProgressText'), (s) => {
        showResults('genResults', s.images || []);
        toast('生成完成');
      });
    } catch (e) { toast('生成失败：' + e.message); }
  });
  $('btnCancelGen').addEventListener('click', () => {
    if (currentTask) { clearInterval(currentTask.timer); apiPost('/api/task/' + currentTask.id + '/cancel', {}); toast('已取消'); }
  });

  function showResults(containerId, images) {
    const c = $(containerId);
    c.innerHTML = '';
    images.forEach((src, i) => {
      const d = document.createElement('div');
      d.className = 'result-card';
      d.innerHTML = `<img src="${src}" alt="result ${i}" /><div class="cap">结果 ${i + 1}</div>`;
      c.appendChild(d);
    });
  }

  // ---- 训练 ----
  $('btnPickTrain').addEventListener('click', async () => {
    if (MOCK) { $('trainData').value = '/demo/dataset'; return; }
    const p = await window.violet.pickFolder();
    if (p) $('trainData').value = p;
  });
  $('btnExportDataset').addEventListener('click', async () => {
    try {
      const r = await apiPost('/api/dataset/export', { source: $('trainData').value || '/demo/dataset', trigger: $('trainTrigger').value });
      trainWizard.goto(1); trainWizard.render('trainWizard');
      toast(`数据集已导出：${r.path || ''}（${r.samples ?? 0} 张）`);
    } catch (e) { toast('导出失败：' + e.message); }
  });
  $('btnTrain').addEventListener('click', async () => {
    const payload = {
      data_dir: $('trainData').value, trigger: $('trainTrigger').value, name: $('trainName').value,
      rank: Number($('trainRank').value), epochs: Number($('trainEpochs').value),
      resolution: Number($('trainRes').value), mem_limit_gb: Number($('trainMem').value),
    };
    if (!payload.data_dir) { toast('请先选择数据集目录'); return; }
    $('trainProgressWrap').classList.remove('hidden');
    progress($('trainProgressBar'), $('trainProgressText'), 1, '提交训练任务…');
    try {
      const r = await apiPost('/api/train/start', payload);
      toast('训练任务已提交（CPU fp32，小时级）：' + (r.task_id || ''));
      if (r.task_id && !MOCK) pollTask(r.task_id, $('trainProgressBar'), $('trainProgressText'), () => toast('训练完成，模型已注册'));
    } catch (e) { toast('训练提交失败：' + e.message); }
  });

  // ---- 模型微调 ----
  async function loadModels() {
    try {
      const r = await apiGet('/api/models');
      $('modelList').innerHTML = (r.models || []).map((m) => `<option>${m}</option>`).join('');
      $('finetuneInfo').innerHTML = `已发现 <b>${(r.models || []).length}</b> 个模型 / LoRA。可调节强度或叠加。`;
    } catch (e) { toast('模型列表加载失败：' + e.message); }
  }
  $('loraScale').addEventListener('input', (e) => { $('loraScaleVal').textContent = (e.target.value / 100).toFixed(2); });
  $('btnLoadModel').addEventListener('click', async () => {
    try { await apiPost('/api/model/load', { name: $('modelList').value, scale: Number($('loraScale').value) / 100 }); toast('模型已加载'); }
    catch (e) { toast('加载失败：' + e.message); }
  });
  $('btnMerge').addEventListener('click', async () => {
    try { await apiPost('/api/model/merge', { name: $('modelList').value, stack: $('loraStack').checked }); toast('已融合并另存为新模型'); }
    catch (e) { toast('融合失败：' + e.message); }
  });

  // ---- 画面调整 ----
  $('btnPickScene').addEventListener('click', async () => {
    if (MOCK) { $('scenePath').value = '/demo/scene.json'; loadLayers(); return; }
    const p = await window.violet.pickFolder();
    if (p) { $('scenePath').value = p; loadLayers(); }
  });
  function loadLayers() {
    $('layerList').innerHTML = `
      <div class="field"><label>图层 1 · 背景　Z 间距</label><input type="range" min="-100" max="100" value="0" class="layerZ" /></div>
      <div class="field"><label>图层 2 · 主体　Z 间距</label><input type="range" min="-100" max="100" value="30" class="layerZ" /></div>
      <div class="field"><label>图层 3 · 前景　Z 间距</label><input type="range" min="-100" max="100" value="60" class="layerZ" /></div>`;
    document.querySelectorAll('.layerZ').forEach((el) => el.addEventListener('input', debounce(renderAdjust, 200)));
  }
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  function lightingPayload() {
    return {
      key: { azimuth: Number($('keyAz').value), elevation: Number($('keyEl').value), intensity: Number($('keyInt').value) / 100 },
      fill: { intensity: Number($('fillInt').value) / 100 },
      ambient: Number($('ambInt').value) / 100,
      ao: Number($('aoInt').value) / 100,
      layers: Array.from(document.querySelectorAll('.layerZ')).map((el, i) => ({ id: i, z: Number(el.value) })),
    };
  }
  async function renderAdjust() {
    if (!$('scenePath').value) { toast('请先选择场景'); return; }
    try {
      const r = await apiPost('/api/render', { scene: $('scenePath').value, lighting: lightingPayload() });
      if (r.image) showResults('adjustResult', [r.image]);
    } catch (e) { toast('渲染失败：' + e.message); }
  }
  $('btnRender').addEventListener('click', renderAdjust);
  ['keyAz','keyEl','keyInt','fillInt','ambInt','aoInt'].forEach((id) => $(id).addEventListener('input', debounce(renderAdjust, 200)));
  $('btnExportImage').addEventListener('click', async () => {
    try { await apiPost('/api/export/image', { scene: $('scenePath').value }); toast('图片已导出'); }
    catch (e) { toast('导出失败：' + e.message); }
  });

  // 初始加载模型列表（若后端就绪）
  setTimeout(() => { if (caps) loadModels(); }, 500);
})();
