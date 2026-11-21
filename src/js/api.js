/** 前端到主进程的 API 封装（走白名单 IPC，不直连网络）。带 mock 回退，便于无后端时预览 UI。 */
const MOCK = typeof window.violet === 'undefined';

const mockCapabilities = {
  cpu: 'Intel Xeon 6983P-C (simulated)',
  physical_cores: 16, logical_cores: 32,
  isa: ['avx2', 'avx512f', 'avx512_bf16', 'amx_bf16', 'amx_int8'],
  gpu: null,
  ram_total_gb: 123, ram_available_gb: 69,
  recommended: { backend: 'openvino', tier: 'balanced', resolution: 1024, steps: 20, threads: 16, quant: 'int8', offload: 'model' },
};

async function apiGet(endpoint) {
  if (MOCK) return mockGet(endpoint);
  const r = await window.violet.get(endpoint);
  if (r.status >= 400) throw new Error(r.body?.detail || '请求失败');
  return r.body;
}
async function apiPost(endpoint, body) {
  if (MOCK) return mockPost(endpoint, body);
  const r = await window.violet.post(endpoint, body);
  if (r.status >= 400) throw new Error(r.body?.detail || '请求失败');
  return r.body;
}

function mockGet(endpoint) {
  if (endpoint === '/api/health') return { status: 'ok', version: '0.1.0' };
  if (endpoint === '/api/capabilities') return mockCapabilities;
  if (endpoint === '/api/models') return { models: ['sd15-base', 'sd15-layerdiffuse', 'my-lora (rank16)'] };
  return {};
}
function mockPost(endpoint, body) {
  if (endpoint === '/api/generate') return { task_id: 'mock-' + Date.now(), images: [] };
  if (endpoint === '/api/render') return { image: 'data:image/png;base64,' };
  if (endpoint === '/api/dataset/export') return { path: '/tmp/mock-dataset', samples: 0 };
  if (endpoint === '/api/train/start') return { task_id: 'mock-train' };
  return { ok: true };
}
