/** preload：通过 contextBridge 暴露最小白名单 API，绝不暴露 ipcRenderer 本身。 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('violet', {
  status: () => ipcRenderer.invoke('app:status'),
  get: (endpoint) => ipcRenderer.invoke('api:get', endpoint),
  post: (endpoint, body) => ipcRenderer.invoke('api:post', endpoint, body),
  pickFolder: () => ipcRenderer.invoke('app:pickFolder'),
  onSidecarReady: (cb) => ipcRenderer.on('sidecar-ready', cb),
});
