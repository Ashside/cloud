import { qs } from '../utils/dom.js';
import { showNotification } from '../ui/notify.js';
import {
  getOutFiles,
  getDatasetFiles,
  pingRemote,
  uploadToRemote,
  uploadDatasetToRemote,
  getTransferStatus,
  startRemoteLora,
  remoteLoraStatus,
  pullRemoteWeight,
} from '../services/apiClient.js';

let initialized = false;
let weightTaskId = null;
let datasetTaskId = null;
let remoteTaskId = null;
let remoteResultFile = null;
let weightPoller = null;
let datasetPoller = null;
let remotePoller = null;

const statusTextMap = {
  pending: '等待中',
  running: '进行中',
  success: '完成',
  error: '失败',
};

function setBar(idPrefix, task) {
  const percent = Math.min(100, Math.max(0, Math.round(task?.progress || 0)));
  const status = statusTextMap[task?.status] || task?.status || '等待';
  const fill = qs(`#${idPrefix}-fill`);
  const text = qs(`#${idPrefix}-status`);
  const num = qs(`#${idPrefix}-progress`);
  if (fill) fill.style.width = `${percent}%`;
  if (num) num.textContent = `${percent}%`;
  if (text) text.textContent = `${status}：${task?.message || ''}`;
}

function setRemoteStatus(task) {
  const percent = Math.min(100, Math.max(0, Math.round(task?.progress || (task?.status === 'success' ? 100 : 0))));
  const status = statusTextMap[task?.status] || task?.status || '等待';
  const fill = qs('#lora-remote-progress-fill');
  const num = qs('#lora-remote-progress-number');
  const text = qs('#lora-remote-status-text');
  const detail = qs('#lora-remote-detail-text');
  if (fill) fill.style.width = `${percent}%`;
  if (num) num.textContent = `${percent}%`;
  if (text) text.textContent = `状态：${status}`;
  if (detail) detail.textContent = task?.message || '';
}

function clearPoller(poller) {
  if (poller) clearInterval(poller);
  return null;
}

async function loadLists() {
  // out files
  const weightSelect = qs('#lora-base-weight');
  const datasetSelect = qs('#lora-dataset');
  if (weightSelect) weightSelect.innerHTML = '<option value="">加载中...</option>';
  if (datasetSelect) datasetSelect.innerHTML = '<option value="">加载中...</option>';
  try {
    const outRes = await getOutFiles();
    const files = outRes.files || [];
    if (weightSelect) {
      weightSelect.innerHTML = '';
      if (!files.length) {
        weightSelect.innerHTML = '<option value="">out 为空</option>';
      } else {
        files.forEach((f) => {
          const opt = document.createElement('option');
          opt.value = f.name;
          opt.textContent = `${f.name} (${formatBytes(f.size)})`;
          weightSelect.appendChild(opt);
        });
      }
    }
  } catch (e) {
    if (weightSelect) weightSelect.innerHTML = '<option value="">加载失败</option>';
  }
  try {
    const dsRes = await getDatasetFiles();
    const files = dsRes.files || [];
    if (datasetSelect) {
      datasetSelect.innerHTML = '';
      if (!files.length) {
        datasetSelect.innerHTML = '<option value="">dataset 为空</option>';
      } else {
        files.forEach((f) => {
          const opt = document.createElement('option');
          opt.value = f.name;
          opt.textContent = `${f.name} (${formatBytes(f.size)})`;
          datasetSelect.appendChild(opt);
        });
      }
    }
  } catch (e) {
    if (datasetSelect) datasetSelect.innerHTML = '<option value="">加载失败</option>';
  }
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / 1024 ** i).toFixed(1)} ${units[i]}`;
}

async function pollTransfer(taskId, prefix) {
  try {
    const task = await getTransferStatus(taskId);
    setBar(prefix, task);
    if (task.status === 'success') {
      showNotification(`${prefix.includes('dataset') ? '数据集' : '权重'}上传完成`);
      return true;
    }
    if (task.status === 'error') {
      showNotification(task.message || '上传失败', 'error');
      return true;
    }
  } catch (error) {
    setBar(prefix, { status: 'error', message: error.message, progress: 0 });
    return true;
  }
  return false;
}

function startTransferPolling(taskId, prefix, setter) {
  setter(clearPoller(setter()));
  pollTransfer(taskId, prefix);
  const poller = setInterval(async () => {
    const done = await pollTransfer(taskId, prefix);
    if (done) setter(clearPoller(poller));
  }, 1200);
  setter(poller);
}

async function handlePing() {
  const url = (qs('#lora-remote-url')?.value || '').trim();
  const token = (qs('#lora-token')?.value || '').trim();
  if (!url) {
    showNotification('请输入远端地址', 'error');
    return;
  }
  setRemoteStatus({ status: 'running', progress: 0, message: 'Ping 远端中...' });
  try {
    const res = await pingRemote({ target_url: url, token });
    if (res.success) {
      showNotification(`Ping 成功：${res.latency_ms} ms`);
      setRemoteStatus({ status: 'success', progress: 10, message: `Ping ${res.latency_ms} ms` });
    } else {
      showNotification(res.error || 'Ping 失败', 'error');
      setRemoteStatus({ status: 'error', progress: 0, message: res.error || 'Ping 失败' });
    }
  } catch (error) {
    showNotification(error.message || 'Ping 失败', 'error');
    setRemoteStatus({ status: 'error', progress: 0, message: error.message || 'Ping 失败' });
  }
}

async function handleStart() {
  const url = (qs('#lora-remote-url')?.value || '').trim();
  const token = (qs('#lora-token')?.value || '').trim();
  const weight = (qs('#lora-base-weight')?.value || '').trim();
  const dataset = (qs('#lora-dataset')?.value || '').trim();
  const overwrite = qs('#lora-overwrite')?.checked || false;
  if (!url || !weight || !dataset) {
    showNotification('请填写远端地址并选择权重与数据集', 'error');
    return;
  }
  // 上传权重
  try {
    const res = await uploadToRemote({ target_url: url, filename: weight, token, overwrite });
    if (!res.task_id) throw new Error(res.error || '上传权重任务创建失败');
    weightTaskId = res.task_id;
    startTransferPolling(weightTaskId, 'lora-upload-weight', (v) => { weightPoller = v; return weightPoller; });
  } catch (error) {
    showNotification(error.message || '上传权重失败', 'error');
    return;
  }

  // 上传数据集
  try {
    const res = await uploadDatasetToRemote({ target_url: url, filename: dataset, token, overwrite });
    if (!res.task_id) throw new Error(res.error || '上传数据集任务创建失败');
    datasetTaskId = res.task_id;
    startTransferPolling(datasetTaskId, 'lora-upload-dataset', (v) => { datasetPoller = v; return datasetPoller; });
  } catch (error) {
    showNotification(error.message || '上传数据集失败', 'error');
    return;
  }

  // 等待上传完成后启动远程 LoRA
  const waitUploads = async () => {
    const statusWeight = await getTransferStatus(weightTaskId);
    const statusDataset = await getTransferStatus(datasetTaskId);
    return statusWeight.status === 'success' && statusDataset.status === 'success';
  };

  const waitLoop = setInterval(async () => {
    try {
      if (await waitUploads()) {
        clearInterval(waitLoop);
        launchRemoteLora(url, token);
      }
    } catch (e) {
      clearInterval(waitLoop);
      showNotification('检查上传状态失败', 'error');
    }
  }, 1500);
}

function currentOrigin() {
  return window.location.origin;
}

async function launchRemoteLora(url, token) {
  const payload = {
    target_url: url,
    token,
    payload: {
      dataset: (qs('#lora-dataset')?.value || '').trim(),
      base_weight: (qs('#lora-base-weight')?.value || '').trim(),
      params: {
        lora_name: (qs('#lora-name')?.value || 'remote_lora').trim(),
        epochs: parseInt(qs('#lora-epochs')?.value || '10', 10),
        batch_size: parseInt(qs('#lora-batch')?.value || '16', 10),
        learning_rate: qs('#lora-lr')?.value || '1e-4',
        hidden_size: parseInt(qs('#lora-hidden')?.value || '512', 10),
        max_seq_len: parseInt(qs('#lora-maxlen')?.value || '512', 10),
        save_interval: parseInt(qs('#lora-save-interval')?.value || '1', 10),
        from_resume: 0,
      },
      callback_url: (qs('#lora-callback-url')?.value || currentOrigin()).trim() || currentOrigin(),
      callback_token: (qs('#lora-callback-token')?.value || '').trim(),
    },
  };
  try {
    const res = await startRemoteLora(payload);
    const data = res.remote_response || {};
    if (!res.success || !data.task_id) {
      showNotification((data.error || res.error || '远程启动失败'), 'error');
      setRemoteStatus({ status: 'error', message: data.error || res.error || '远程启动失败' });
      return;
    }
    remoteTaskId = data.task_id;
    remoteResultFile = data.result_file;
    const taskInput = qs('#lora-remote-task');
    const resultInput = qs('#lora-remote-result');
    if (taskInput) taskInput.value = remoteTaskId;
    if (resultInput) resultInput.value = remoteResultFile || '';
    showNotification('远程 LoRA 训练已启动');
    startRemotePolling();
  } catch (error) {
    showNotification(error.message || '远程启动失败', 'error');
    setRemoteStatus({ status: 'error', message: error.message || '远程启动失败' });
  }
}

async function pollRemoteOnce() {
  if (!remoteTaskId) {
    showNotification('没有远程任务ID', 'error');
    return true;
  }
  const url = (qs('#lora-remote-url')?.value || '').trim();
  const token = (qs('#lora-token')?.value || '').trim();
  try {
    const res = await remoteLoraStatus({ target_url: url, task_id: remoteTaskId, token });
    if (!res.success) {
      setRemoteStatus({ status: 'error', message: res.error || `远程状态查询失败(${res.status_code || ''})` });
      return true;
    }
    const task = res.data || {};
    setRemoteStatus(task);
    if (task.status === 'success') {
      showNotification('远程训练完成');
      remoteResultFile = task.result_file || remoteResultFile;
      const resultInput = qs('#lora-remote-result');
      if (resultInput && remoteResultFile) resultInput.value = remoteResultFile;
      return true;
    }
    if (task.status === 'error') {
      showNotification(task.message || '远程训练失败', 'error');
      return true;
    }
  } catch (error) {
    setRemoteStatus({ status: 'error', message: error.message || '远程查询失败' });
    return true;
  }
  return false;
}

function startRemotePolling() {
  remotePoller = clearPoller(remotePoller);
  pollRemoteOnce();
  remotePoller = setInterval(async () => {
    const done = await pollRemoteOnce();
    if (done) remotePoller = clearPoller(remotePoller);
  }, 2000);
}

async function handleManualDownload() {
  if (!remoteResultFile || !remoteTaskId) {
    showNotification('缺少远程结果信息', 'error');
    return;
  }
  const url = (qs('#lora-remote-url')?.value || '').trim();
  const token = (qs('#lora-token')?.value || '').trim();
  try {
    const res = await pullRemoteWeight({
      source_url: url,
      filename: remoteResultFile,
      token,
      overwrite: true,
    });
    if (!res.task_id) throw new Error(res.error || '下载任务创建失败');
    showNotification('开始从 B 拉取 LoRA 权重');
  } catch (error) {
    showNotification(error.message || '拉取失败', 'error');
  }
}

export function initLoraExchange() {
  if (initialized) return;
  initialized = true;
  const btnRefresh = qs('#btn-refresh-lora-lists');
  const btnPing = qs('#btn-lora-ping');
  const btnStart = qs('#btn-start-remote-lora');
  const btnPollRemote = qs('#btn-poll-remote-lora');
  const btnDownload = qs('#btn-download-lora');
  if (btnRefresh) btnRefresh.addEventListener('click', loadLists);
  if (btnPing) btnPing.addEventListener('click', handlePing);
  if (btnStart) btnStart.addEventListener('click', handleStart);
  if (btnPollRemote) btnPollRemote.addEventListener('click', () => {
    pollRemoteOnce();
  });
  if (btnDownload) btnDownload.addEventListener('click', handleManualDownload);
  const cbUrl = qs('#lora-callback-url');
  if (cbUrl && !cbUrl.value) cbUrl.value = currentOrigin();
  loadLists();
}

export function onEnterLora() {
  loadLists();
  if (remoteTaskId) startRemotePolling();
}
