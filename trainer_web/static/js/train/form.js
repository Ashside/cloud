import { startTrain, getDefaults } from '../services/apiClient.js';
import { showNotification } from '../ui/notify.js';

const fallbackDefaults = {
  pretrain: { save_dir: '../out', save_weight: 'pretrain', epochs: 1, batch_size: 32, learning_rate: 5e-4, data_path: '../dataset/pretrain_hq.jsonl', from_weight: 'none', log_interval: 100, save_interval: 100, hidden_size: 512, num_hidden_layers: 8, max_seq_len: 512, use_moe: 0 },
  sft: { save_dir: '../out', save_weight: 'full_sft', epochs: 2, batch_size: 16, learning_rate: 5e-7, data_path: '../dataset/sft_mini_512.jsonl', from_weight: 'pretrain', log_interval: 100, save_interval: 100, hidden_size: 512, num_hidden_layers: 8, max_seq_len: 512, use_moe: 0 },
  lora: { save_dir: '../out/lora', lora_name: 'lora_identity', epochs: 50, batch_size: 32, learning_rate: 1e-4, data_path: '../dataset/lora_identity.jsonl', from_weight: 'full_sft', log_interval: 10, save_interval: 1, hidden_size: 512, num_hidden_layers: 8, max_seq_len: 512, use_moe: 0 },
  dpo: { save_dir: '../out', save_weight: 'dpo', epochs: 1, batch_size: 4, learning_rate: 4e-8, data_path: '../dataset/dpo.jsonl', from_weight: 'full_sft', log_interval: 100, save_interval: 100, beta: 0.1, hidden_size: 512, num_hidden_layers: 8, max_seq_len: 1024, use_moe: 0 },
  ppo: { save_dir: '../out', save_weight: 'ppo_actor', epochs: 1, batch_size: 2, learning_rate: 8e-8, data_path: '../dataset/rlaif-mini.jsonl', log_interval: 1, save_interval: 10, clip_epsilon: 0.1, vf_coef: 0.5, kl_coef: 0.02, reasoning: 1, update_old_actor_freq: 4, reward_model_path: '../../internlm2-1_8b-reward', hidden_size: 512, num_hidden_layers: 8, max_seq_len: 66, use_moe: 0 },
  grpo: { save_dir: '../out', save_weight: 'grpo', epochs: 1, batch_size: 2, learning_rate: 8e-8, data_path: '../dataset/rlaif-mini.jsonl', log_interval: 1, save_interval: 10, beta: 0.02, num_generations: 8, reasoning: 1, reward_model_path: '../../internlm2-1_8b-reward', hidden_size: 512, num_hidden_layers: 8, max_seq_len: 66, use_moe: 0 },
  spo: { save_dir: '../out', save_weight: 'spo', epochs: 1, batch_size: 2, learning_rate: 1e-7, data_path: '../dataset/rlaif-mini.jsonl', log_interval: 1, save_interval: 10, beta: 0.02, reasoning: 1, reward_model_path: '../../internlm2-1_8b-reward', hidden_size: 512, num_hidden_layers: 8, max_seq_len: 66, use_moe: 0 },
};

let defaults = { ...fallbackDefaults };
let resourceDefaults = {};
let currentType = 'pretrain';

export async function initTrainForm() {
  await loadDefaultsFromServer();
  bindTypeCards();
  initPaneSwitcher();
  initGpuSelectors();
  const form = document.getElementById('train-form');
  if (form) form.addEventListener('submit', onSubmit);
  toggleFieldsForType(currentType);
  applyDefaults(currentType);
  return { resourceDefaults };
}

async function loadDefaultsFromServer() {
  try {
    const res = await getDefaults();
    defaults = res.train_types || defaults;
    resourceDefaults = res.resources || {};
    if (res.gpu) {
      window.hasGpu = res.gpu.has_gpu;
      window.gpuCount = res.gpu.count;
    }
  } catch (err) {
    showNotification('获取默认参数失败，使用内置默认值', 'warning');
  }
}

function bindTypeCards() {
  const cards = document.querySelectorAll('.type-card');
  cards.forEach((card) => {
    card.addEventListener('click', () => {
      cards.forEach((c) => c.classList.remove('active'));
      card.classList.add('active');
      const type = card.dataset.type;
      currentType = type;
      const hiddenInput = document.getElementById('train_type');
      if (hiddenInput) hiddenInput.value = type;
      toggleFieldsForType(type);
      applyDefaults(type);
    });
  });
}

function toggleFieldsForType(type) {
  const pretrainSft = document.querySelectorAll('.pretrain-sft');
  const fromWeightFields = document.querySelectorAll('.from-weight');
  const loraFields = document.querySelectorAll('.lora');
  const rlCards = document.querySelectorAll('.rl-card');

  pretrainSft.forEach((el) => setVisibility(el, ['pretrain', 'sft', 'dpo', 'ppo', 'grpo', 'spo'].includes(type)));
  fromWeightFields.forEach((el) => setVisibility(el, !['ppo', 'grpo', 'spo'].includes(type)));
  loraFields.forEach((el) => setVisibility(el, type === 'lora'));

  rlCards.forEach((card) => {
    const active = card.dataset.for === type;
    setVisibility(card, active);
    card.querySelectorAll('input, select, textarea').forEach((el) => {
      el.disabled = !active;
    });
  });
}

function setVisibility(el, visible) {
  if (!el) return;
  el.style.display = visible ? 'block' : 'none';
  el.querySelectorAll?.('input, select, textarea').forEach((inp) => {
    inp.disabled = !visible;
  });
}

function applyDefaults(type) {
  const config = defaults[type] || {};
  Object.entries(config).forEach(([name, val]) => {
    const nodes = document.querySelectorAll(`[name="${name}"]`);
    nodes.forEach((node) => {
      if (node.type === 'checkbox') {
        node.checked = String(val) === '1' || val === true || val === 1;
      } else {
        node.value = val;
      }
    });
  });
}

function initPaneSwitcher() {
  window.switchTrainPane = (pane) => {
    const panes = document.querySelectorAll('.train-pane');
    panes.forEach((p) => p.classList.add('hidden'));
    const target = document.getElementById(`pane-${pane}`);
    if (target) target.classList.remove('hidden');
    const tabs = document.querySelectorAll('.pane-tab');
    tabs.forEach((t) => t.classList.remove('active'));
    const activeTab = document.querySelector(`.pane-tab[data-pane="${pane}"]`);
    activeTab?.classList.add('active');
  };
}

function initGpuSelectors() {
  const hasGpu = window.hasGpu === true;
  const gpuCount = Number(window.gpuCount || 0);
  const modeSel = document.getElementById('training_mode');
  const single = document.getElementById('single-gpu-selection');
  const multi = document.getElementById('multi-gpu-selection');
  if (!modeSel) return;
  function updateVisibility() {
    const mode = modeSel.value;
    if (single) single.style.display = mode === 'single_gpu' ? 'block' : 'none';
    if (multi) multi.style.display = mode === 'multi_gpu' ? 'block' : 'none';
  }
  if (!hasGpu) {
    modeSel.value = 'cpu';
    if (single) single.style.display = 'none';
    if (multi) multi.style.display = 'none';
  } else {
    const gpuNumInput = document.getElementById('gpu_num');
    if (gpuNumInput && gpuCount > 0) gpuNumInput.value = gpuCount;
  }
  updateVisibility();
  modeSel.addEventListener('change', updateVisibility);
}

function onSubmit(e) {
  e.preventDefault();
  const form = e.currentTarget;
  const data = {};
  const trainingModeSel = form.querySelector('#training_mode');
  const trainingMode = trainingModeSel ? trainingModeSel.value : 'cpu';
  const inputs = form.querySelectorAll('input, select, textarea');
  inputs.forEach((el) => {
    const name = el.name;
    if (!name || name === 'training_mode' || el.disabled) return;
    let value = el.value;
    if (el.type === 'checkbox') {
      value = el.checked ? el.value || '1' : '0';
    }
    if (name === 'gpu_num') {
      const multi = document.getElementById('multi-gpu-selection');
      if (!(multi && multi.style.display !== 'none')) return;
    }
    if (name === 'device') {
      if (trainingMode === 'single_gpu') value = `cuda:${value}`;
      else if (trainingMode === 'cpu') value = 'cpu';
      else return;
    }
    data[name] = value;
  });
  data.train_type = currentType;
  showNotification('正在启动训练...', 'info');
  startTrain(data)
    .then((result) => {
      if (result.success) {
        showNotification('训练已开始！', 'success');
        setTimeout(() => {
          const processTab = document.querySelector('.tab[onclick*="processes"]');
          if (processTab) processTab.click();
        }, 600);
      } else showNotification('训练启动失败：' + result.error, 'error');
    })
    .catch(() => {
      showNotification('启动训练中，请耐心等待...', 'info');
    });
}
