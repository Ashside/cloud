import { getDatasetFiles, downloadDataset, downloadWeights, uploadWeights } from '../services/apiClient.js';
import { showNotification } from '../ui/notify.js';

let datasetCache = [];

export function initResourceTools(resourceDefaults = {}) {
  setupDatasetPanel(resourceDefaults);
  setupWeightPanel(resourceDefaults);
}

function setupDatasetPanel(resourceDefaults) {
  const datasetIdInput = document.getElementById('dataset_id');
  const datasetDirInput = document.getElementById('dataset_dir');
  if (datasetIdInput && resourceDefaults.modelscope_dataset) datasetIdInput.value = resourceDefaults.modelscope_dataset;
  if (datasetDirInput && resourceDefaults.dataset_dir) datasetDirInput.value = resourceDefaults.dataset_dir;

  const loadBtn = document.getElementById('load-dataset-files');
  const downloadBtn = document.getElementById('download-dataset');
  loadBtn?.addEventListener('click', async () => {
    const datasetId = datasetIdInput?.value || resourceDefaults.modelscope_dataset;
    const token = document.getElementById('dataset_token')?.value || '';
    showNotification('正在获取文件列表...', 'info');
    try {
      const res = await getDatasetFiles({ dataset_id: datasetId, token });
      datasetCache = res.files || [];
      renderDatasetFiles(datasetCache, res.source);
      showNotification('文件列表已更新', 'success');
    } catch (err) {
      showNotification('获取文件列表失败', 'error');
    }
  });

  downloadBtn?.addEventListener('click', async () => {
    const selected = Array.from(document.querySelectorAll('.dataset-file input[type="checkbox"]:checked')).map((el) => el.value);
    const datasetId = datasetIdInput?.value || resourceDefaults.modelscope_dataset;
    const token = document.getElementById('dataset_token')?.value || '';
    const destDir = datasetDirInput?.value || resourceDefaults.dataset_dir;
    if (!datasetId) {
      showNotification('请填写数据集地址', 'error');
      return;
    }
    const files = selected.length > 0 ? selected : datasetCache.map((f) => f.name);
    showNotification(`开始下载 ${files.length} 个文件...`, 'info');
    try {
      const res = await downloadDataset({ dataset_id: datasetId, token, dest_dir: destDir, files });
      if (res.success) showNotification(`下载完成，保存在 ${res.dest}`, 'success');
      else showNotification(res.message || '下载失败', 'error');
    } catch (err) {
      showNotification('下载失败，请检查命令行可用性', 'error');
    }
  });

  if (resourceDefaults.modelscope_files && resourceDefaults.modelscope_files.length > 0) {
    datasetCache = resourceDefaults.modelscope_files.map((name) => ({ name }));
    renderDatasetFiles(datasetCache, 'env');
  }
}

function renderDatasetFiles(files, source = 'env') {
  const container = document.getElementById('dataset-files');
  if (!container) return;
  if (!files || files.length === 0) {
    container.textContent = '未找到文件';
    return;
  }
  const list = files
    .map((f) => {
      const label = f.size ? `${f.name} (${f.size})` : f.name;
      return `<label class="dataset-file"><input type="checkbox" value="${f.name}" checked> ${label}</label>`;
    })
    .join('');
  container.innerHTML = `<div class="file-source">来源: ${source}</div>${list}`;
}

function setupWeightPanel(resourceDefaults) {
  const providerSel = document.getElementById('weight_provider');
  const repoInput = document.getElementById('repo_id');
  const tokenInput = document.getElementById('weight_token');
  const localPathInput = document.getElementById('local_weight_path');
  const remotePathInput = document.getElementById('remote_weight_path');
  const targetDirInput = document.getElementById('download_target_dir');

  if (localPathInput && resourceDefaults.upload_path) localPathInput.value = resourceDefaults.upload_path;
  if (targetDirInput && resourceDefaults.download_path) targetDirInput.value = resourceDefaults.download_path;

  const downloadBtn = document.getElementById('download-weights-btn');
  const uploadBtn = document.getElementById('upload-weights-btn');

  downloadBtn?.addEventListener('click', async () => {
    const payload = {
      provider: providerSel?.value || 'ms',
      repo_id: repoInput?.value || '',
      token: tokenInput?.value || '',
      remote_path: remotePathInput?.value || '',
      target_dir: targetDirInput?.value || '',
    };
    if (!payload.repo_id) {
      showNotification('请填写项目地址', 'error');
      return;
    }
    showNotification('开始下载权重...', 'info');
    try {
      const res = await downloadWeights(payload);
      if (res.success) showNotification(`下载完成: ${res.target_dir}`, 'success');
      else showNotification(res.message || '下载失败', 'error');
    } catch (err) {
      showNotification('下载失败，请检查命令行工具是否可用', 'error');
    }
  });

  uploadBtn?.addEventListener('click', async () => {
    const payload = {
      provider: providerSel?.value || 'ms',
      repo_id: repoInput?.value || '',
      token: tokenInput?.value || '',
      local_path: localPathInput?.value || '',
      remote_path: remotePathInput?.value || '',
    };
    if (!payload.repo_id || !payload.local_path) {
      showNotification('请填写项目地址和本地路径', 'error');
      return;
    }
    showNotification('开始上传权重...', 'info');
    try {
      const res = await uploadWeights(payload);
      if (res.success) showNotification('上传完成', 'success');
      else showNotification(res.message || '上传失败', 'error');
    } catch (err) {
      showNotification('上传失败，请检查命令行工具是否可用', 'error');
    }
  });
}
