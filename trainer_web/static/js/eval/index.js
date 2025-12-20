import { qs } from '../utils/dom.js';
import { showNotification } from '../ui/notify.js';
import { getEvalWeights, runEval } from '../services/apiClient.js';

let initialized = false;

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / 1024 ** i).toFixed(1)} ${units[i]}`;
}

async function loadWeights() {
  const baseSel = qs('#eval-weight');
  const loraSel = qs('#eval-lora');
  if (baseSel) baseSel.innerHTML = '<option value="">加载中...</option>';
  if (loraSel) loraSel.innerHTML = '<option value="None">不使用 LoRA</option>';
  try {
    const data = await getEvalWeights();
    const baseWeights = data.base_weights || [];
    const loraWeights = data.lora_weights || [];
    if (baseSel) {
      baseSel.innerHTML = '';
      if (!baseWeights.length) {
        baseSel.innerHTML = '<option value="">out 目录暂无权重</option>';
      } else {
        baseWeights.forEach((w) => {
          const opt = document.createElement('option');
          opt.value = w.name;
          opt.textContent = `${w.name}_${w.hidden_size}${w.use_moe ? '_moe' : ''}.pth`;
          opt.dataset.hidden = w.hidden_size;
          opt.dataset.useMoe = w.use_moe;
          baseSel.appendChild(opt);
        });
        // 选中第一个时同步 hidden_size/use_moe
        if (baseWeights[0]) {
          setHiddenAndMoe(baseWeights[0].hidden_size, baseWeights[0].use_moe);
        }
      }
    }
    if (loraSel) {
      loraWeights.forEach((w) => {
        const opt = document.createElement('option');
        opt.value = w.name;
        opt.textContent = `${w.name}_${w.hidden_size}${w.use_moe ? '_moe' : ''}.pth`;
        loraSel.appendChild(opt);
      });
    }
  } catch (error) {
    if (baseSel) baseSel.innerHTML = '<option value="">加载失败</option>';
    showNotification(error.message || '加载权重失败', 'error');
  }
}

function setHiddenAndMoe(hidden, useMoe) {
  const hiddenInput = qs('#eval-hidden');
  const moeCheckbox = qs('#eval-use-moe');
  if (hiddenInput && hidden) hiddenInput.value = hidden;
  if (moeCheckbox) moeCheckbox.checked = Boolean(useMoe);
}

function bindWeightChange() {
  const baseSel = qs('#eval-weight');
  if (!baseSel) return;
  baseSel.addEventListener('change', (e) => {
    const opt = e.target.selectedOptions[0];
    if (!opt) return;
    const hidden = opt.dataset.hidden ? Number(opt.dataset.hidden) : null;
    const useMoe = opt.dataset.useMoe === '1';
    if (hidden) setHiddenAndMoe(hidden, useMoe);
  });
}

function renderResults(outputs) {
  const container = qs('#eval-results');
  if (!container) return;
  if (!outputs || !outputs.length) {
    container.textContent = '暂无输出';
    return;
  }
  container.innerHTML = '';
  outputs.forEach((item, idx) => {
    const block = document.createElement('div');
    block.className = 'log-block';
    block.innerHTML = `
      <div class="log-header">样例 ${idx + 1}</div>
      <div class="log-line"><strong>👶 Prompt:</strong> ${item.prompt}</div>
      <div class="log-line"><strong>🤖 回复:</strong> ${item.response}</div>
    `;
    container.appendChild(block);
  });
}

async function handleRun() {
  const payload = {
    load_from: (qs('#eval-load-from')?.value || 'model').trim(),
    save_dir: 'out',
    weight: (qs('#eval-weight')?.value || '').trim(),
    lora_weight: (qs('#eval-lora')?.value || 'None').trim(),
    hidden_size: Number(qs('#eval-hidden')?.value || '512'),
    num_hidden_layers: Number(qs('#eval-layers')?.value || '8'),
    use_moe: qs('#eval-use-moe')?.checked ? 1 : 0,
    max_new_tokens: Number(qs('#eval-max-tokens')?.value || '512'),
    temperature: Number(qs('#eval-temp')?.value || '0.85'),
    top_p: Number(qs('#eval-top-p')?.value || '0.85'),
    historys: Number(qs('#eval-historys')?.value || '0'),
    inference_rope_scaling: false,
    prompt: (qs('#eval-prompt')?.value || '').trim(),
    use_default_prompts: qs('#eval-use-default')?.checked || false,
  };
  if (!payload.weight) {
    showNotification('请先选择 out/ 权重', 'error');
    return;
  }
  if (!payload.use_default_prompts && !payload.prompt) {
    showNotification('请输入 prompt 或勾选内置测试', 'error');
    return;
  }
  renderResults([{ prompt: '运行中...', response: '请稍候' }]);
  try {
    const res = await runEval(payload);
    if (!res.success) {
      showNotification(res.error || '评测失败', 'error');
      renderResults([{ prompt: '错误', response: res.error || '评测失败' }]);
      return;
    }
    renderResults(res.outputs || []);
  } catch (error) {
    showNotification(error.message || '评测失败', 'error');
    renderResults([{ prompt: '错误', response: error.message || '评测失败' }]);
  }
}

export function initEval() {
  if (initialized) return;
  initialized = true;
  const btnRun = qs('#btn-run-eval');
  const btnRefresh = qs('#btn-refresh-eval-weights');
  if (btnRun) btnRun.addEventListener('click', handleRun);
  if (btnRefresh) btnRefresh.addEventListener('click', loadWeights);
  bindWeightChange();
  loadWeights();
}

export function onEnterEval() {
  loadWeights();
}
