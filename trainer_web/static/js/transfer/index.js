import { qs } from '../utils/dom.js';
import { showNotification } from '../ui/notify.js';
import {
  getOutFiles,
  getRemoteOutFiles,
  getTransferStatus,
  pingRemote,
  pullRemoteWeight,
  uploadToRemote,
} from '../services/apiClient.js';

let initialized = false;
let uploadPoller = null;
let downloadPoller = null;
let uploadTaskId = null;
let downloadTaskId = null;

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / 1024 ** i).toFixed(1)} ${units[i]}`;
}

function setProgress(type, task) {
  const fill = qs(`#${type}-progress-fill`);
  const number = qs(`#${type}-progress-number`);
  const statusText = qs(`#${type}-status-text`);
  const detail = qs(`#${type}-detail-text`);

  const percent = Math.min(100, Math.max(0, Math.round(task?.progress || 0)));
  const statusMap = {
    pending: '等待中',
    running: '进行中',
    success: '完成',
    error: '失败',
    'ping': 'Ping中',
    'ping-ok': 'Ping成功',
  };
  const statusLabel = statusMap[task?.status] || task?.status || '等待任务';
  if (fill) fill.style.width = `${percent}%`;
  if (number) number.textContent = `${percent}%`;
  if (statusText) statusText.textContent = `状态：${statusLabel}`;
  if (detail) detail.textContent = task?.message || '';
}

function clearPoller(poller) {
  if (poller) clearInterval(poller);
  return null;
}

async function loadLocalOutFiles() {
  const select = qs('#upload-file-select');
  if (!select) return;
  select.innerHTML = '<option value="">加载中...</option>';
  try {
    const data = await getOutFiles();
    const files = data.files || [];
    const base = data.base || '';
    const baseNode = qs('#local-out-base');
    if (baseNode) baseNode.textContent = base;
    if (!files.length) {
      select.innerHTML = '<option value="">out目录暂无文件</option>';
      return;
    }
    select.innerHTML = '';
    files.forEach((f) => {
      const opt = document.createElement('option');
      opt.value = f.name;
      opt.textContent = `${f.name} (${formatBytes(f.size)})`;
      select.appendChild(opt);
    });
  } catch (error) {
    select.innerHTML = '<option value="">加载失败</option>';
    showNotification(`读取out目录失败：${error.message || error}`, 'error');
  }
}

async function handlePing() {
  const url = (qs('#upload-remote-url')?.value || '').trim();
  const token = (qs('#upload-token')?.value || '').trim();
  if (!url) {
    showNotification('请输入目标服务器地址', 'error');
    return;
  }
  setProgress('upload', { status: 'ping', progress: 0, message: '正在Ping远端...' });
  try {
    const res = await pingRemote({ target_url: url, token });
    if (res.success) {
      const detail = `延迟 ${res.latency_ms} ms`;
      setProgress('upload', { status: 'ping-ok', progress: 0, message: detail });
      showNotification(`Ping成功：${detail}`);
    } else {
      const msg = res.error || 'Ping失败';
      setProgress('upload', { status: 'error', progress: 0, message: msg });
      showNotification(msg, 'error');
    }
  } catch (error) {
    const msg = error.message || 'Ping失败';
    setProgress('upload', { status: 'error', progress: 0, message: msg });
    showNotification(msg, 'error');
  }
}

async function handleUpload() {
  const url = (qs('#upload-remote-url')?.value || '').trim();
  const filename = (qs('#upload-file-select')?.value || '').trim();
  const token = (qs('#upload-token')?.value || '').trim();
  const overwrite = qs('#upload-overwrite')?.checked || false;
  if (!url || !filename) {
    showNotification('请填写目标地址并选择out目录下的文件', 'error');
    return;
  }
  setProgress('upload', { status: 'running', progress: 0, message: '正在创建上传任务...' });
  try {
    const res = await uploadToRemote({ target_url: url, filename, token, overwrite });
    if (!res.task_id) {
      const msg = res.error || '任务创建失败';
      setProgress('upload', { status: 'error', progress: 0, message: msg });
      showNotification(msg, 'error');
      return;
    }
    uploadTaskId = res.task_id;
    showNotification('上传任务已开始');
    startUploadPolling(res.task_id);
  } catch (error) {
    const msg = error.message || '上传任务创建失败';
    setProgress('upload', { status: 'error', progress: 0, message: msg });
    showNotification(msg, 'error');
  }
}

async function handleLoadRemoteList() {
  const url = (qs('#download-remote-url')?.value || '').trim();
  const token = (qs('#download-token')?.value || '').trim();
  const select = qs('#download-file-select');
  if (!url || !select) {
    showNotification('请填写来源服务器地址', 'error');
    return;
  }
  select.innerHTML = '<option value="">加载中...</option>';
  try {
    const res = await getRemoteOutFiles({ target_url: url, token });
    if (!res.success) {
      const msg = res.error || '无法获取远程文件列表';
      select.innerHTML = '<option value="">加载失败</option>';
      showNotification(msg, 'error');
      return;
    }
    const files = res.data?.files || [];
    if (!files.length) {
      select.innerHTML = '<option value="">远程out目录为空</option>';
      return;
    }
    select.innerHTML = '';
    files.forEach((f) => {
      const opt = document.createElement('option');
      opt.value = f.name;
      opt.textContent = `${f.name} (${formatBytes(f.size)})`;
      select.appendChild(opt);
    });
  } catch (error) {
    select.innerHTML = '<option value="">加载失败</option>';
    showNotification(error.message || '加载远程列表失败', 'error');
  }
}

async function handleDownload() {
  const url = (qs('#download-remote-url')?.value || '').trim();
  const filename = (qs('#download-file-select')?.value || '').trim();
  const token = (qs('#download-token')?.value || '').trim();
  const overwrite = qs('#download-overwrite')?.checked || false;
  if (!url || !filename) {
    showNotification('请选择远程文件并填写来源服务器地址', 'error');
    return;
  }
  setProgress('download', { status: 'running', progress: 0, message: '正在创建下载任务...' });
  try {
    const res = await pullRemoteWeight({ source_url: url, filename, token, overwrite });
    if (!res.task_id) {
      const msg = res.error || '任务创建失败';
      setProgress('download', { status: 'error', progress: 0, message: msg });
      showNotification(msg, 'error');
      return;
    }
    downloadTaskId = res.task_id;
    showNotification('下载任务已开始');
    startDownloadPolling(res.task_id);
  } catch (error) {
    const msg = error.message || '下载任务创建失败';
    setProgress('download', { status: 'error', progress: 0, message: msg });
    showNotification(msg, 'error');
  }
}

async function pollTask(taskId, type) {
  try {
    const task = await getTransferStatus(taskId);
    setProgress(type, task);
    if (task.status === 'success') {
      showNotification(`${type === 'upload' ? '上传' : '下载'}完成`);
      return true;
    }
    if (task.status === 'error') {
      showNotification(task.message || '任务失败', 'error');
      return true;
    }
  } catch (error) {
    setProgress(type, { status: 'error', progress: 0, message: error.message || '获取进度失败' });
    return true;
  }
  return false;
}

function startUploadPolling(taskId) {
  uploadPoller = clearPoller(uploadPoller);
  pollTask(taskId, 'upload');
  uploadPoller = setInterval(async () => {
    const done = await pollTask(taskId, 'upload');
    if (done) uploadPoller = clearPoller(uploadPoller);
  }, 1200);
}

function startDownloadPolling(taskId) {
  downloadPoller = clearPoller(downloadPoller);
  pollTask(taskId, 'download');
  downloadPoller = setInterval(async () => {
    const done = await pollTask(taskId, 'download');
    if (done) downloadPoller = clearPoller(downloadPoller);
  }, 1200);
}

export function initTransfer() {
  if (initialized) return;
  initialized = true;
  const pingBtn = qs('#btn-ping-remote');
  const uploadBtn = qs('#btn-start-upload');
  const refreshLocalBtn = qs('#btn-refresh-out');
  const loadRemoteListBtn = qs('#btn-load-remote-list');
  const downloadBtn = qs('#btn-start-download');

  if (pingBtn) pingBtn.addEventListener('click', handlePing);
  if (uploadBtn) uploadBtn.addEventListener('click', handleUpload);
  if (refreshLocalBtn) refreshLocalBtn.addEventListener('click', loadLocalOutFiles);
  if (loadRemoteListBtn) loadRemoteListBtn.addEventListener('click', handleLoadRemoteList);
  if (downloadBtn) downloadBtn.addEventListener('click', handleDownload);

  loadLocalOutFiles();
}

export function onEnterTransfer() {
  loadLocalOutFiles();
  if (uploadTaskId) startUploadPolling(uploadTaskId);
  if (downloadTaskId) startDownloadPolling(downloadTaskId);
}
