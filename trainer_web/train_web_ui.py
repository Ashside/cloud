import os
import sys
import subprocess
import threading
import json
import socket
import atexit
import signal
import re
import uuid
import urllib.parse
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from flask import g
import time
import psutil
import glob
import pathlib
import requests
from werkzeug.utils import secure_filename
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# 复用eval逻辑
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed

# 尝试导入torch来检测GPU
try:
    import torch
    HAS_TORCH = True
    # 检测可用的GPU数量和设备信息
    if torch.cuda.is_available():
        GPU_COUNT = torch.cuda.device_count()
        # 获取GPU设备名称
        GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]
    else:
        GPU_COUNT = 0
        GPU_NAMES = []
except ImportError:
    HAS_TORCH = False
    GPU_COUNT = 0
    GPU_NAMES = []

def calculate_training_progress(process_id, process_info):
    """
    计算训练进度信息
    从日志文件中提取训练进度、loss、epoch等信息
    """
    progress = {
        'percentage': 0,
        'current_epoch': 0,
        'total_epochs': 0,
        'current_step': 0,
        'total_steps': 0,
        'remaining_time': '计算中...',
        'current_loss': None,
        'current_lr': None
    }
    
    # 如果进程不在运行且没有日志文件，返回空进度
    if not process_info.get('running', False):
        # 检查是否有日志文件，如果有则继续解析
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, '../logfile')
        log_dir = os.path.abspath(log_dir)
        
        log_file_exists = False
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                if filename.endswith(f'{process_id}.log'):
                    log_file_exists = True
                    break
        
        # 如果没有日志文件且进程不在运行，返回空进度
        if not log_file_exists:
            return progress
    
    try:
        # 获取日志文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, '../logfile')
        log_dir = os.path.abspath(log_dir)
        
        log_file = None
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                if filename.endswith(f'{process_id}.log'):
                    log_file = os.path.join(log_dir, filename)
                    break
        
        if not log_file or not os.path.exists(log_file):
            return progress
        
        # 读取日志文件的最后1000行
        def read_last_lines(file_path, n=1000):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # 使用更高效的方法读取最后n行
                    lines = []
                    for line in f:
                        lines.append(line.strip())
                        if len(lines) > n:
                            lines.pop(0)
                    return lines
            except Exception:
                return []
        
        lines = read_last_lines(log_file, 1000)
        
        # 从日志中提取进度信息
        current_epoch = 0
        total_epochs = 0
        current_loss = None
        current_lr = None
        
        for line in reversed(lines):  # 从最新日志开始
            line = line.strip()
            if not line:
                continue
                
            # 提取epoch信息 - 支持多种格式
            if not total_epochs:
                # 格式: epoch 3/10, Epoch 3 of 10, [3/10], 第3轮/共10轮, Epoch:[1/1]
                epoch_patterns = [
                    r'Epoch:\[(\d+)/(\d+)\]',                      # Epoch:[1/1] - 新格式
                    r'epoch\s+(\d+)\s*/\s*(\d+)',
                    r'Epoch\s+(\d+)\s*of\s*(\d+)',
                    r'\[(\d+)/(\d+)\]',
                    r'epoch\s*[:：]\s*(\d+)\s*/\s*(\d+)',
                    r'第\s*(\d+)\s*轮\s*/\s*共\s*(\d+)\s*轮'
                ]
                
                for pattern in epoch_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        if r'Epoch:\[' in pattern:
                            current_epoch = int(match.group(1))
                            total_epochs = int(match.group(2))
                        else:
                            current_epoch = int(match.group(1))
                            total_epochs = int(match.group(2))
                        break
            
            # 提取step信息 - 支持多种格式
            # 格式: (74/44160), step 150/1000, Step 150 of 1000, [150/1000], step: 150/1000
            step_patterns = [
                r'\((\d+)/(\d+)\)',                            # (74/44160) - 新格式
                r'step\s+(\d+)\s*/\s*(\d+)',
                r'Step\s+(\d+)\s*of\s*(\d+)',
                r'\[(\d+)/(\d+)\]',
                r'step\s*[:：]\s*(\d+)\s*/\s*(\d+)',
                r'第\s*(\d+)\s*步\s*/\s*共\s*(\d+)\s*步',
                r'步数\s*(\d+)\s*/\s*(\d+)',
                r'batch\s+(\d+)\s*/\s*(\d+)',  # 也支持batch格式
                r'Batch\s+(\d+)\s*of\s*(\d+)'
            ]
            
            for pattern in step_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    progress['current_step'] = int(match.group(1))
                    progress['total_steps'] = int(match.group(2))
                    break
            
            # 提取loss信息 - 支持多种格式
            if not current_loss:
                # 格式: loss:8.896761, loss: 4.32, training_loss: 4.32, train_loss: 4.32, Loss: 4.32, 训练损失: 4.32
                loss_patterns = [
                    r'loss:([\d.]+(?:e[+-]?\d+)?)',                    # loss:8.896761 - 新格式
                    r'loss[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',           # loss: 4.32
                    r'training_loss[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',  # training_loss: 4.32
                    r'train_loss[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',     # train_loss: 4.32
                    r'Loss[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',          # Loss: 4.32
                    r'训练损失[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',        # 训练损失: 4.32
                    r'损失[\s:=]\s*([\d.]+(?:e[+-]?\d+)?)',           # 损失: 4.32
                    r'\s+([\d.]+(?:e[+-]?\d+)?)\s*loss',             # 4.32 loss
                    r'\s+([\d.]+(?:e[+-]?\d+)?)\s*训练损失',           # 4.32 训练损失
                    r'(?:loss|损失|training_loss|train_loss)\s*=\s*([\d.]+(?:e[+-]?\d+)?)'  # loss = 4.32
                ]
                
                for pattern in loss_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        # 取最后一个匹配的loss值
                        loss_value = float(matches[-1])
                        if 0 < loss_value < 100:  # 合理的loss范围
                            current_loss = loss_value
                            break
            
            # 提取学习率信息 - 支持多种格式
            if not current_lr:
                # 格式: lr:0.000549999999, lr: 1e-4, learning_rate: 1e-4, LR: 1e-4, 学习率: 1e-4
                lr_patterns = [
                    r'lr:([\d.e+-]+)',                              # lr:0.000549999999 - 新格式
                    r'lr[\s:=]\s*([\d.e+-]+)',
                    r'learning_rate[\s:=]\s*([\d.e+-]+)',
                    r'LR[\s:=]\s*([\d.e+-]+)',
                    r'学习率[\s:=]\s*([\d.e+-]+)'
                ]
                
                for pattern in lr_patterns:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        # 取最后一个匹配的lr值
                        lr_value = float(matches[-1])
                        if 0 < lr_value < 1:  # 合理的lr范围
                            current_lr = f"{lr_value:.2e}"
                            break
            
            # 如果已经收集到足够信息，提前退出
            if total_epochs and current_loss and current_lr:
                break
        
        # 计算进度百分比 - 支持epoch和step双重进度
        percentage = 0
        if total_epochs > 0:
            # 基础epoch进度
            epoch_percentage = (current_epoch / total_epochs) * 100
            
            # 如果有step信息，在当前epoch内计算step进度
            if progress['total_steps'] > 0 and progress['current_step'] > 0:
                # 计算当前epoch内的step进度
                step_percentage_in_epoch = (progress['current_step'] / progress['total_steps']) * 100
                # 将step进度加到epoch进度上（每个epoch占总进度的1/total_epochs）
                step_contribution = step_percentage_in_epoch / total_epochs
                percentage = min(100, max(0, int(epoch_percentage + step_contribution)))
            else:
                # 只有epoch信息的传统计算方式
                percentage = min(100, max(0, int(epoch_percentage)))
        
        # 更新进度字典
        progress['percentage'] = percentage
        progress['current_epoch'] = current_epoch
        progress['total_epochs'] = total_epochs
        progress['current_loss'] = current_loss
        progress['current_lr'] = current_lr
        
        # 估算剩余时间（增强计算）
        remaining_time = '计算中...'
        if current_epoch > 0 and total_epochs > current_epoch:
            # 从日志中提取时间信息
            for line in reversed(lines):
                # 格式: remaining: 1:30:45, ETA: 1:30:45, 预计剩余: 1小时30分钟, epoch_Time:332.0min:
                time_patterns = [
                    r'epoch_Time:([\d.]+)min:',                    # epoch_Time:332.0min: - 新格式
                    r'remaining[\s:=]\s*(\d+):(\d+):(\d+)',      # remaining: 1:30:45
                    r'ETA[\s:=]\s*(\d+):(\d+):(\d+)',            # ETA: 1:30:45
                    r'预计剩余[\s:=]\s*(\d+)[\s小时]*[\s:]?(\d+)?[\s分钟]*',  # 预计剩余: 1小时30分钟
                    r'剩余时间[\s:=]\s*(\d+)[\s小时]*[\s:]?(\d+)?[\s分钟]*',  # 剩余时间: 1小时30分钟
                    r'time left[\s:=]\s*(\d+)[\s:]?(\d+)?[\s:]?(\d+)?',  # time left: 1:30:45
                    r'还需[\s:=]\s*(\d+)[\s小时]*[\s:]?(\d+)?[\s分钟]*'  # 还需: 1小时30分钟
                ]
                
                for pattern in time_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        # 处理epoch_Time格式
                        if 'epoch_Time:' in pattern:
                            minutes = float(match.group(1))
                            if minutes > 0:
                                if minutes >= 60:
                                    hours = int(minutes // 60)
                                    remaining_minutes = int(minutes % 60)
                                    if hours > 0:
                                        remaining_time = f"{hours}小时{remaining_minutes}分钟"
                                    else:
                                        remaining_time = f"{remaining_minutes}分钟"
                                else:
                                    remaining_time = f"{int(minutes)}分钟"
                                break
                        else:
                            groups = match.groups()
                            if len(groups) >= 3 and all(groups[:3]):
                                # 小时:分钟:秒格式
                                hours = int(groups[0])
                                minutes = int(groups[1])
                                seconds = int(groups[2])
                                if hours > 0 or minutes > 0 or seconds > 0:
                                    parts = []
                                    if hours > 0: parts.append(f"{hours}小时")
                                    if minutes > 0: parts.append(f"{minutes}分钟")
                                    if seconds > 0 and hours == 0 and minutes == 0:
                                        parts.append(f"{seconds}秒")
                                    remaining_time = ''.join(parts)
                                    break
                            elif len(groups) >= 2:
                                # 小时和分钟格式
                                hours = int(groups[0])
                                minutes = int(groups[1]) if groups[1] else 0
                                if hours > 0 or minutes > 0:
                                    parts = []
                                    if hours > 0: parts.append(f"{hours}小时")
                                    if minutes > 0: parts.append(f"{minutes}分钟")
                                    remaining_time = ''.join(parts)
                                    break
                
                if remaining_time != '计算中...':
                    break
            
            # 如果没有找到时间信息，根据进度估算
            if remaining_time == '计算中...':
                # 假设每epoch时间大致相同
                elapsed_time = time.time() - process_info.get('start_timestamp', time.time())
                if current_epoch > 0:
                    time_per_epoch = elapsed_time / current_epoch
                    remaining_epochs = total_epochs - current_epoch
                    remaining_seconds = remaining_epochs * time_per_epoch
                    
                    if remaining_seconds > 3600:
                        remaining_time = f"{remaining_seconds / 3600:.1f}小时"
                    elif remaining_seconds > 60:
                        remaining_time = f"{remaining_seconds / 60:.1f}分钟"
                    else:
                        remaining_time = f"{int(remaining_seconds)}秒"
        
        return {
            'percentage': percentage,
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
            'current_step': progress['current_step'],
            'total_steps': progress['total_steps'],
            'remaining_time': remaining_time,
            'current_loss': f"{current_loss:.4f}" if current_loss else None,
            'current_lr': current_lr
        }
        
    except Exception as e:
        print(f"计算进度时出错: {e}")
        return progress

# 训练方式支持检测
def get_supported_training_methods():
    """获取当前环境支持的训练方法"""
    methods = {
        'pretrain': True,  # 预训练总是支持
        'sft': True,       # SFT总是支持
        'lora': True,      # LoRA总是支持
        'dpo': True,       # DPO总是支持
        'multi_gpu': HAS_TORCH and GPU_COUNT > 1  # 多GPU训练需要PyTorch和多个GPU
    }
    return methods

# 获取当前环境支持的训练方法
SUPPORTED_METHODS = get_supported_training_methods()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__, template_folder='templates', static_folder='static')

# 存储训练进程的信息
training_processes = {}

# 进程信息持久化文件
PROCESSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_processes.json')

# PID文件
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_web_ui.pid')

# 项目与传输相关的路径/配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
OUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'out'))
DATASET_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'dataset'))
TRANSFER_TOKEN = os.environ.get('TRANSFER_TOKEN')  # 可选：用于简单鉴权
ALLOWED_PORTS = [6006, 6008]  # autodl 仅开放端口
TRANSFER_DEBUG = os.environ.get('TRANSFER_DEBUG', '0') == '1'

# 跨服务器传输任务状态
transfer_tasks = {}
transfer_lock = threading.Lock()

# LoRA 远程协作任务状态
lora_exchange_tasks = {}
lora_lock = threading.Lock()


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def ensure_dataset_dir():
    os.makedirs(DATASET_DIR, exist_ok=True)
    return DATASET_DIR


def normalize_target_url(raw_url):
    """标准化目标URL，缺失协议时默认http"""
    if not raw_url:
        return None
    if isinstance(raw_url, str):
        raw_url = raw_url.strip()
    if not raw_url:
        return None
    parsed = urllib.parse.urlparse(raw_url)
    if not parsed.scheme:
        raw_url = f"http://{raw_url}"
        parsed = urllib.parse.urlparse(raw_url)
    if not parsed.netloc:
        return None
    # autodl 仅开放 6006/6008，若未指定端口则默认 6006
    if parsed.port is None and parsed.hostname:
        default_port = ALLOWED_PORTS[0]
        netloc = f"{parsed.hostname}:{default_port}"
        parsed = parsed._replace(netloc=netloc)
    # 仅保留 scheme://netloc 以及 path（去掉末尾的斜杠）
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip('/')
    return normalized


def safe_out_path(rel_path):
    """将相对路径限制在out目录下，并返回(安全相对路径, 绝对路径)"""
    if not rel_path:
        return None, None
    normalized = os.path.normpath(rel_path).replace('\\', '/')
    normalized = normalized.lstrip('/')
    if normalized.startswith('..'):
        return None, None
    parts = [secure_filename(p) for p in normalized.split('/') if p not in ('', '.')]
    safe_rel = '/'.join([p for p in parts if p])
    if not safe_rel:
        return None, None
    ensure_out_dir()
    abs_path = os.path.abspath(os.path.join(OUT_DIR, *safe_rel.split('/')))
    if not abs_path.startswith(OUT_DIR):
        return None, None
    return safe_rel, abs_path


def safe_dataset_path(rel_path):
    """将相对路径限制在dataset目录下，并返回(安全相对路径, 绝对路径)"""
    if not rel_path:
        return None, None
    normalized = os.path.normpath(rel_path).replace('\\', '/')
    normalized = normalized.lstrip('/')
    if normalized.startswith('..'):
        return None, None
    parts = [secure_filename(p) for p in normalized.split('/') if p not in ('', '.')]
    safe_rel = '/'.join([p for p in parts if p])
    if not safe_rel:
        return None, None
    ensure_dataset_dir()
    abs_path = os.path.abspath(os.path.join(DATASET_DIR, *safe_rel.split('/')))
    if not abs_path.startswith(DATASET_DIR):
        return None, None
    return safe_rel, abs_path


def parse_weight_name(filename):
    """解析权重文件名，返回前缀/hidden_size/moe标记"""
    base = os.path.basename(filename)
    match = re.match(r'(.+?)_(\d+)(?:_moe)?\.pth$', base)
    if not match:
        return None
    name = match.group(1)
    hidden = int(match.group(2))
    is_moe = base.endswith('_moe.pth')
    return {'name': name, 'hidden_size': hidden, 'use_moe': 1 if is_moe else 0}


def list_weights():
    ensure_out_dir()
    base_weights = []
    for root, _, files in os.walk(OUT_DIR):
        # 跳过LoRA子目录
        if os.path.basename(root) == 'lora':
            continue
        for f in files:
            if not f.endswith('.pth'):
                continue
            info = parse_weight_name(f)
            if not info:
                continue
            rel = os.path.relpath(os.path.join(root, f), OUT_DIR)
            base_weights.append({**info, 'relative_path': rel})
    base_weights.sort(key=lambda x: x['relative_path'])

    lora_dir = os.path.join(OUT_DIR, 'lora')
    lora_weights = []
    if os.path.exists(lora_dir):
        for f in os.listdir(lora_dir):
            if not f.endswith('.pth'):
                continue
            info = parse_weight_name(f)
            if not info:
                continue
            lora_weights.append({**info, 'relative_path': os.path.join('lora', f)})
        lora_weights.sort(key=lambda x: x['relative_path'])

    return base_weights, lora_weights


def check_transfer_token(req, allow_body=False):
    """当设置了TRANSFER_TOKEN时，校验传输令牌"""
    if not TRANSFER_TOKEN:
        return None
    provided = req.headers.get('X-Transfer-Token') or req.args.get('token')
    if not provided and allow_body and req.is_json:
        provided = (req.get_json(silent=True) or {}).get('token')
    if provided != TRANSFER_TOKEN:
        return jsonify({'error': '无效的传输令牌'}), 401
    return None


def create_transfer_task(task_type, filename, endpoint=None):
    task_id = uuid.uuid4().hex
    with transfer_lock:
        transfer_tasks[task_id] = {
            'id': task_id,
            'type': task_type,
            'filename': filename,
            'endpoint': endpoint,
            'status': 'pending',
            'progress': 0,
            'message': '待开始'
        }
        # 保留最近的任务，避免无限增长
        if len(transfer_tasks) > 100:
            for key in list(transfer_tasks.keys())[:-80]:
                transfer_tasks.pop(key, None)
    return task_id


def update_transfer_task(task_id, **updates):
    with transfer_lock:
        if task_id in transfer_tasks:
            transfer_tasks[task_id].update(updates)


def get_transfer_task(task_id):
    with transfer_lock:
        return transfer_tasks.get(task_id)


def create_lora_task(payload):
    task_id = uuid.uuid4().hex
    with lora_lock:
        lora_exchange_tasks[task_id] = {
            'id': task_id,
            'status': 'pending',
            'message': '等待启动',
            'process_id': None,
            'result_file': None,
            'callback': payload.get('callback'),
            'created_at': int(time.time())
        }
    return task_id


def update_lora_task(task_id, **updates):
    with lora_lock:
        if task_id in lora_exchange_tasks:
            lora_exchange_tasks[task_id].update(updates)


def get_lora_task(task_id):
    with lora_lock:
        return lora_exchange_tasks.get(task_id)

# Authentication removed - allow anonymous training

# 启动训练进程
def start_training_process(train_type, params, client_id=None):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用详细的时间戳作为进程ID和日志文件名
    process_id = time.strftime('%Y%m%d_%H%M%S')
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    log_file = os.path.join(log_dir, f"train_{train_type}_{process_id}.log")
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取GPU数量参数，如果存在且大于1，则使用torchrun启动多卡训练
    gpu_num = int(params.get('gpu_num', 0)) if 'gpu_num' in params else 0
    use_torchrun = HAS_TORCH and GPU_COUNT > 0 and gpu_num > 1
    
    try:
        from .dispatcher import build_command
    except ImportError:
        import sys as _sys
        import os as _os
        _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
        from dispatcher import build_command
    cmd = build_command(train_type, params, gpu_num, use_torchrun)
    if cmd is None:
        return None
    
    # 创建日志文件
    with open(log_file, 'w') as f:
        f.write(f"开始训练 {train_type} 进程\n")
        f.write(f"命令: {' '.join(cmd)}\n\n")
    
    # 启动进程
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    # 存储进程信息
    training_processes[process_id] = {
        'process': process,
        'train_type': train_type,
        'log_file': log_file,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'start_timestamp': time.time(),  # 添加时间戳用于进度计算
        'running': True,
        'error': False,
        'train_monitor': params.get('train_monitor', 'none'),  # 保存训练监控设置
        'swanlab_url': None,
        'next_line_is_swanlab_url': False,
        'client_id': client_id
    }
    
    # 开始读取输出
    def read_output():
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # 检查是否是swanlab链接的行
                    output_stripped = output.strip()
                    if training_processes[process_id]['next_line_is_swanlab_url']:
                        # 保存swanlab链接
                        training_processes[process_id]['swanlab_url'] = output_stripped
                        training_processes[process_id]['next_line_is_swanlab_url'] = False
                    elif 'swanlab: 🚀 View run at' in output_stripped:
                        # 标记下一行是swanlab链接
                        training_processes[process_id]['next_line_is_swanlab_url'] = True
                    
                    with open(log_file, 'a') as f:
                        f.write(output)
            # 检查进程是否成功结束
            if process.returncode != 0:
                training_processes[process_id]['error'] = True
        finally:
            training_processes[process_id]['running'] = False
    
    # 启动线程读取输出
    threading.Thread(target=read_output, daemon=True).start()
    
    return process_id

# Flask路由
@app.route('/')
def index():
    # 传递GPU信息到前端
    return render_template(
        'index.html',
        has_gpu=HAS_TORCH and GPU_COUNT > 0,
        gpu_count=GPU_COUNT,
        transfer_debug=TRANSFER_DEBUG
    )

@app.route('/healthz')
def healthz():
    try:
        return jsonify({'status': 'ok', 'gpu': GPU_COUNT, 'methods': SUPPORTED_METHODS}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    train_type = data.get('train_type')
    
    # 移除不相关的参数
    params = data.copy()
    
    # 处理复选框参数
    if 'from_resume' not in params:
        params['from_resume'] = '0'
    
    # 启动训练进程 - 允许匿名训练，不传入client_id
    process_id = start_training_process(train_type, params)
    
    if process_id:
        return jsonify({'success': True, 'process_id': process_id})
    else:
        return jsonify({'success': False, 'error': '无效的训练类型'})

# 测试端点 - 添加模拟训练进程
@app.route('/test/add_process', methods=['POST'])
def add_test_process():
    """添加一个测试进程用于验证自动更新功能"""
    import subprocess
    import threading
    
    process_id = f"test_process_{int(time.time())}"
    
    # 创建测试训练命令 - 包含step进度和新的log格式
    test_command = [
        'python', '-c', '''
import time
import sys

print("2024-11-21 14:30:00 - Starting pretrain training")
sys.stdout.flush()
time.sleep(1)

print("2024-11-21 14:30:01 - Loading dataset from ../dataset/pretrain_hq.jsonl")
sys.stdout.flush()
time.sleep(1)

print("2024-11-21 14:30:02 - Model initialized with 108M parameters")
sys.stdout.flush()
time.sleep(2)

# 测试单epoch但多step的情况，使用新的log格式
print("2024-11-21 14:30:03 - Epoch:[1/1] Starting training")
sys.stdout.flush()
time.sleep(1)

total_steps = 20
for step in range(1, total_steps + 1):
    # 模拟step进度，使用新的格式
    if step % 5 == 0 or step == total_steps:
        print(f"2024-11-21 14:30:{4 + step} - Epoch:[1/1]({step}/{total_steps}) Processing")
        sys.stdout.flush()
    
    # 模拟训练过程，使用新的格式
    loss = 4.5 - step * 0.1
    lr = 1e-4 * (0.95 ** step)
    if step % 3 == 0:
        print(f"2024-11-21 14:30:{4 + step} - Epoch:[1/1]({step}/{total_steps}) loss:{loss:.6f} lr:{lr:.2e} epoch_Time:{step * 5.5:.1f}min:")
        sys.stdout.flush()
    
    time.sleep(0.5)

print("2024-11-21 14:30:25 - Training completed successfully")
sys.stdout.flush()
        '''
    ]
    
    # 启动进程
    process = subprocess.Popen(
        test_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 保存进程信息
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logfile')
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    training_processes[process_id] = {
        'process': process,
        'train_type': 'pretrain',
        'log_file': os.path.join(log_dir, f'{process_id}.log'),
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'start_timestamp': time.time(),
        'running': True,
        'error': False,
        'train_monitor': 'none',
        'swanlab_url': None
    }
    
    # 启动线程读取输出并写入日志文件
    def read_output():
        try:
            log_file = training_processes[process_id]['log_file']
            with open(log_file, 'w') as f:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        f.write(line)
                        f.flush()
            process.wait()
            training_processes[process_id]['running'] = False
            if process.returncode != 0:
                training_processes[process_id]['error'] = True
        except Exception as e:
            print(f"读取测试进程输出时出错: {e}")
            training_processes[process_id]['running'] = False
            training_processes[process_id]['error'] = True
    
    threading.Thread(target=read_output, daemon=True).start()
    
    return jsonify({
        'success': True,
        'process_id': process_id,
        'message': '测试进程已添加'
    })

@app.route('/processes')
def processes():
    result = []
    for process_id, info in training_processes.items():
        # 确定状态
        status = '运行中' if info['running'] else \
                '手动停止' if 'manually_stopped' in info and info['manually_stopped'] else \
                '出错' if info['error'] else '已完成'
        
        # 计算训练进度信息
        progress = calculate_training_progress(process_id, info)
                
        result.append({
            'id': process_id,
            'train_type': info['train_type'],
            'start_time': info['start_time'],
            'running': info['running'],
            'error': info['error'],
            'status': status,
            'train_monitor': info.get('train_monitor', 'none'),  # 添加train_monitor字段
            'swanlab_url': info.get('swanlab_url'),  # 添加swanlab_url字段
            'progress': progress  # 添加进度信息
        })
    return jsonify(result)

@app.route('/api/browse')
def browse_files():
    """
    浏览服务器文件系统
    支持远程文件选择功能
    """
    try:
        # 获取请求的路径参数
        path = request.args.get('path', './')
        
        # 安全检查：限制访问范围
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        
        # 解析请求的路径
        if path.startswith('./'):
            # 相对路径，基于项目根目录
            full_path = os.path.abspath(os.path.join(project_root, path[2:]))
        elif path.startswith('/'):
            # 绝对路径，检查是否在项目目录内
            full_path = os.path.abspath(path)
        else:
            # 相对路径，基于项目根目录
            full_path = os.path.abspath(os.path.join(project_root, path))
        
        # 安全检查：确保路径在项目目录内
        if not full_path.startswith(project_root):
            full_path = project_root
        
        # 检查路径是否存在
        if not os.path.exists(full_path):
            return jsonify({'error': '路径不存在', 'path': path})
        
        # 获取目录内容
        if os.path.isdir(full_path):
            items = []
            try:
                # 列出目录内容
                for item in sorted(os.listdir(full_path)):
                    item_path = os.path.join(full_path, item)
                    
                    # 跳过隐藏文件和系统文件
                    if item.startswith('.') or item.startswith('__'):
                        continue
                    
                    try:
                        stat = os.stat(item_path)
                        items.append({
                            'name': item,
                            'path': item_path,  # 返回绝对路径
                            'relative_path': os.path.relpath(item_path, project_root),  # 同时返回相对路径用于显示
                            'type': 'directory' if os.path.isdir(item_path) else 'file',
                            'size': stat.st_size if os.path.isfile(item_path) else 0,
                            'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                        })
                    except (OSError, PermissionError):
                        # 跳过无法访问的项目
                        continue
                
                return jsonify({
                    'current_path': full_path,  # 返回绝对路径
                    'relative_path': os.path.relpath(full_path, project_root),  # 相对路径用于显示
                    'absolute_path': full_path,
                    'items': items,
                    'parent': os.path.dirname(full_path) if full_path != project_root else None
                })
            except (OSError, PermissionError) as e:
                return jsonify({'error': f'无法访问目录: {str(e)}', 'path': path})
        
        else:
            # 如果是文件，返回文件信息
            stat = os.stat(full_path)
            return jsonify({
                'name': os.path.basename(full_path),
                'path': full_path,  # 返回绝对路径
                'relative_path': os.path.relpath(full_path, project_root),  # 相对路径用于显示
                'type': 'file',
                'size': stat.st_size,
                'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            })
            
    except Exception as e:
        return jsonify({'error': f'浏览文件时出错: {str(e)}'})

@app.route('/api/quick-paths')
def quick_paths():
    """
    返回常用路径快捷方式
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        
        quick_paths = [
            {'name': '项目根目录', 'path': './', 'type': 'directory'},
            {'name': '数据集目录', 'path': './dataset', 'type': 'directory'},
            {'name': '模型检查点', 'path': './checkpoints', 'type': 'directory'},
            {'name': '日志文件', 'path': './logfile', 'type': 'directory'}
        ]
        
        # 验证路径是否存在
        valid_paths = []
        for item in quick_paths:
            full_path = os.path.join(project_root, item['path'][2:] if item['path'].startswith('./') else item['path'])
            if os.path.exists(full_path):
                valid_paths.append(item)
        
        return jsonify({'paths': valid_paths})
        
    except Exception as e:
        return jsonify({'error': f'获取快捷路径时出错: {str(e)}'})


@app.route('/api/ping')
def api_ping():
    token_error = check_transfer_token(request)
    if token_error:
        return token_error
    return jsonify({
        'status': 'ok',
        'server': socket.gethostname(),
        'time': int(time.time())
    })


@app.route('/api/out-files')
def list_out_files():
    """列出out目录下可传输的文件"""
    token_error = check_transfer_token(request)
    if token_error:
        return token_error
    ensure_out_dir()
    files = []
    for root, _, filenames in os.walk(OUT_DIR):
        for name in filenames:
            file_path = os.path.join(root, name)
            try:
                stat = os.stat(file_path)
                rel_path = os.path.relpath(file_path, OUT_DIR)
                files.append({
                    'name': rel_path,
                    'size': stat.st_size,
                    'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                })
            except OSError:
                continue
    files.sort(key=lambda x: x['name'])
    return jsonify({'base': OUT_DIR, 'files': files})


@app.route('/api/eval/weights')
def api_eval_weights():
    base_weights, lora_weights = list_weights()
    return jsonify({'base_weights': base_weights, 'lora_weights': lora_weights, 'out_dir': OUT_DIR})


@app.route('/api/dataset-files')
def list_dataset_files():
    """列出dataset目录下可传输的数据文件"""
    token_error = check_transfer_token(request)
    if token_error:
        return token_error
    ensure_dataset_dir()
    files = []
    for root, _, filenames in os.walk(DATASET_DIR):
        for name in filenames:
            file_path = os.path.join(root, name)
            try:
                stat = os.stat(file_path)
                rel_path = os.path.relpath(file_path, DATASET_DIR)
                files.append({
                    'name': rel_path,
                    'size': stat.st_size,
                    'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                })
            except OSError:
                continue
    files.sort(key=lambda x: x['name'])
    return jsonify({'base': DATASET_DIR, 'files': files})


@app.route('/api/transfer-status/<task_id>')
def transfer_status(task_id):
    task = get_transfer_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(task)


@app.route('/api/ping-remote', methods=['POST'])
def ping_remote():
    data = request.get_json(silent=True) or {}
    target_url = normalize_target_url(data.get('target_url'))
    token = data.get('token')
    if not target_url:
        return jsonify({'error': '目标地址不能为空'}), 400
    headers = {}
    if token:
        headers['X-Transfer-Token'] = token
    start = time.time()
    try:
        resp = requests.get(f"{target_url}/api/ping", headers=headers, timeout=5)
        latency = int((time.time() - start) * 1000)
        if resp.status_code != 200:
            return jsonify({'error': f'远程返回{resp.status_code}', 'latency_ms': latency}), 502
        payload = {}
        try:
            payload = resp.json()
        except Exception:
            payload = {'raw': resp.text[:200]}
        return jsonify({'success': True, 'latency_ms': latency, 'remote': target_url, 'data': payload})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 502


@app.route('/api/remote-out-files', methods=['POST'])
def remote_out_files():
    data = request.get_json(silent=True) or {}
    target_url = normalize_target_url(data.get('target_url'))
    token = data.get('token')
    if not target_url:
        return jsonify({'error': '目标地址不能为空'}), 400
    headers = {}
    if token:
        headers['X-Transfer-Token'] = token
    try:
        resp = requests.get(f"{target_url}/api/out-files", headers=headers, timeout=10)
        if resp.status_code != 200:
            return jsonify({'error': f'远程返回{resp.status_code}', 'remote': target_url}), 502
        return jsonify({'success': True, 'remote': target_url, 'data': resp.json()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 502


@app.route('/api/upload-dataset-to-remote', methods=['POST'])
def upload_dataset_to_remote():
    data = request.get_json(silent=True) or {}
    target_url = normalize_target_url(data.get('target_url'))
    filename = data.get('filename')
    token = data.get('token')
    overwrite = str(data.get('overwrite', False)).lower() in ('1', 'true', 'yes', 'on')
    if not target_url or not filename:
        return jsonify({'error': '目标地址和数据文件均不能为空'}), 400
    safe_rel, file_path = safe_dataset_path(filename)
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '请选择dataset目录下存在的文件'}), 400

    task_id = create_transfer_task('upload-dataset', safe_rel, target_url)
    threading.Thread(
        target=upload_worker_dataset,
        args=(task_id, target_url, safe_rel, file_path, token, overwrite),
        daemon=True
    ).start()
    return jsonify({'task_id': task_id})


@app.route('/api/upload-to-remote', methods=['POST'])
def upload_to_remote():
    data = request.get_json(silent=True) or {}
    target_url = normalize_target_url(data.get('target_url'))
    filename = data.get('filename')
    token = data.get('token')
    overwrite = str(data.get('overwrite', False)).lower() in ('1', 'true', 'yes', 'on')
    if not target_url or not filename:
        return jsonify({'error': '目标地址和文件名均不能为空'}), 400
    safe_rel, file_path = safe_out_path(filename)
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '请选择out目录下存在的文件'}), 400

    task_id = create_transfer_task('upload', safe_rel, target_url)
    threading.Thread(
        target=upload_worker,
        args=(task_id, target_url, safe_rel, file_path, token, overwrite),
        daemon=True
    ).start()
    return jsonify({'task_id': task_id})


@app.route('/api/receive-weight', methods=['POST'])
def receive_weight():
    token_error = check_transfer_token(request)
    if token_error:
        return token_error
    filename = request.headers.get('X-Filename') or request.args.get('filename')
    overwrite = request.args.get('overwrite', '0') in ('1', 'true', 'yes')
    safe_rel, dest_path = safe_out_path(filename)
    if not dest_path:
        return jsonify({'success': False, 'error': '文件名无效或越界'}), 400
    ensure_out_dir()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and not overwrite:
        return jsonify({'success': False, 'error': '文件已存在，请开启覆盖后再试'}), 409

    bytes_written = 0
    try:
        with open(dest_path, 'wb') as f:
            for chunk in iter(lambda: request.stream.read(1024 * 1024), b''):
                if not chunk:
                    break
                f.write(chunk)
                bytes_written += len(chunk)
        return jsonify({'success': True, 'saved_as': safe_rel, 'bytes': bytes_written})
    except Exception as e:
        print(f"接收文件失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/receive-dataset', methods=['POST'])
def receive_dataset():
    token_error = check_transfer_token(request)
    if token_error:
        return token_error
    filename = request.headers.get('X-Filename') or request.args.get('filename')
    overwrite = request.args.get('overwrite', '0') in ('1', 'true', 'yes')
    safe_rel, dest_path = safe_dataset_path(filename)
    if not dest_path:
        return jsonify({'success': False, 'error': '文件名无效或越界'}), 400
    ensure_dataset_dir()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and not overwrite:
        return jsonify({'success': False, 'error': '文件已存在，请开启覆盖后再试'}), 409

    bytes_written = 0
    try:
        with open(dest_path, 'wb') as f:
            for chunk in iter(lambda: request.stream.read(1024 * 1024), b''):
                if not chunk:
                    break
                f.write(chunk)
                bytes_written += len(chunk)
        return jsonify({'success': True, 'saved_as': safe_rel, 'bytes': bytes_written})
    except Exception as e:
        print(f"接收数据集失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download-weight')
def download_weight():
    token_error = check_transfer_token(request)
    if token_error:
        return token_error
    filename = request.args.get('filename')
    safe_rel, file_path = safe_out_path(filename)
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '文件不存在'}), 404
    return send_file(file_path, as_attachment=True, download_name=os.path.basename(file_path))


@app.route('/api/pull-remote-weight', methods=['POST'])
def pull_remote_weight():
    data = request.get_json(silent=True) or {}
    source_url = normalize_target_url(data.get('source_url'))
    filename = data.get('filename')
    token = data.get('token')
    overwrite = str(data.get('overwrite', False)).lower() in ('1', 'true', 'yes', 'on')
    if not source_url or not filename:
        return jsonify({'error': '来源地址和文件名均不能为空'}), 400
    safe_rel, dest_path = safe_out_path(filename)
    if not dest_path:
        return jsonify({'error': '无效的文件名'}), 400

    task_id = create_transfer_task('download', safe_rel, source_url)
    threading.Thread(
        target=pull_worker,
        args=(task_id, source_url, safe_rel, dest_path, token, overwrite),
        daemon=True
    ).start()
    return jsonify({'task_id': task_id})


@app.route('/api/lora/start', methods=['POST'])
def lora_start():
    """在当前服务器启动LoRA训练（供远程调用）"""
    token_error = check_transfer_token(request, allow_body=True)
    if token_error:
        return token_error
    data = request.get_json(silent=True) or {}
    dataset_name = data.get('dataset')
    base_weight = data.get('base_weight')
    params = data.get('params') or {}
    callback_url = normalize_target_url(data.get('callback_url'))
    callback_token = data.get('callback_token')

    safe_dataset, dataset_path = safe_dataset_path(dataset_name)
    safe_weight, weight_path = safe_out_path(base_weight)
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'error': '数据集不存在或无效'}), 400
    if not weight_path or not os.path.exists(weight_path):
        return jsonify({'error': '基座权重不存在或无效'}), 400

    lora_name = params.get('lora_name') or 'remote_lora'
    hidden_size = int(params.get('hidden_size') or 512)
    # 结果文件名推测
    result_file = os.path.join(OUT_DIR, 'lora', f"{lora_name}_{hidden_size}.pth")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    # 构造训练参数
    train_params = {
        'train_type': 'lora',
        'data_path': dataset_path,
        'from_weight': params.get('from_weight') or derive_from_weight_prefix(safe_weight),
        'lora_name': lora_name,
        'save_dir': os.path.join(OUT_DIR, 'lora'),
        'epochs': params.get('epochs') or 10,
        'batch_size': params.get('batch_size') or 16,
        'learning_rate': params.get('learning_rate') or 1e-4,
        'hidden_size': hidden_size,
        'num_hidden_layers': params.get('num_hidden_layers') or 8,
        'max_seq_len': params.get('max_seq_len') or 512,
        'use_moe': params.get('use_moe') or 0,
        'log_interval': params.get('log_interval') or 10,
        'save_interval': params.get('save_interval') or 1,
        'from_resume': params.get('from_resume') or 0,
        'train_monitor': 'none'
    }

    process_id = start_training_process('lora', train_params)
    if not process_id:
        return jsonify({'error': '无法启动LoRA训练'}), 500

    task_id = create_lora_task({'callback': {'url': callback_url, 'token': callback_token}})
    update_lora_task(task_id, status='running', message='LoRA训练中', process_id=process_id, result_file=os.path.relpath(result_file, OUT_DIR))
    threading.Thread(
        target=watch_lora_training,
        args=(task_id, process_id, result_file, {'url': callback_url, 'token': callback_token}),
        daemon=True
    ).start()

    return jsonify({
        'task_id': task_id,
        'process_id': process_id,
        'result_file': os.path.relpath(result_file, OUT_DIR)
    })


@app.route('/api/lora/status/<task_id>')
def lora_status(task_id):
    token_error = check_transfer_token(request)
    if token_error:
        return token_error
    task = get_lora_task(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(task)


@app.route('/api/lora/remote-start', methods=['POST'])
def lora_remote_start():
    data = request.get_json(silent=True) or {}
    target_url = normalize_target_url(data.get('target_url'))
    token = data.get('token')
    payload = data.get('payload') or {}
    if not target_url:
        return jsonify({'error': '目标地址不能为空'}), 400
    headers = {}
    if token:
        headers['X-Transfer-Token'] = token
    try:
        resp = requests.post(f"{target_url}/api/lora/start", json=payload, headers=headers, timeout=30)
        return jsonify({'success': resp.status_code == 200, 'remote_response': resp.json() if resp.content else {}, 'status_code': resp.status_code})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 502


@app.route('/api/lora/remote-status', methods=['POST'])
def lora_remote_status():
    data = request.get_json(silent=True) or {}
    target_url = normalize_target_url(data.get('target_url'))
    task_id = data.get('task_id')
    token = data.get('token')
    if not target_url or not task_id:
        return jsonify({'error': '缺少目标地址或任务ID'}), 400
    headers = {}
    if token:
        headers['X-Transfer-Token'] = token
    try:
        resp = requests.get(f"{target_url}/api/lora/status/{task_id}", headers=headers, timeout=15)
        if resp.status_code != 200:
            return jsonify({'success': False, 'status_code': resp.status_code, 'remote': target_url})
        return jsonify({'success': True, 'data': resp.json()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 502


def upload_worker(task_id, target_url, safe_rel, file_path, token=None, overwrite=False):
    update_transfer_task(task_id, status='running', progress=0, message='准备上传')
    try:
        file_size = os.path.getsize(file_path)
        headers = {
            'Content-Type': 'application/octet-stream',
            'X-Filename': os.path.basename(safe_rel)
        }
        if token:
            headers['X-Transfer-Token'] = token
        params = {'overwrite': '1' if overwrite else '0'}
        bytes_sent = 0

        def stream_file():
            nonlocal bytes_sent
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    bytes_sent += len(chunk)
                    if file_size:
                        progress = min(100, round(bytes_sent / file_size * 100, 2))
                    else:
                        progress = 100
                    update_transfer_task(task_id, status='running', progress=progress, message='上传中...')
                    yield chunk

        resp = requests.post(
            f"{target_url}/api/receive-weight",
            params=params,
            data=stream_file(),
            headers=headers,
            timeout=600
        )
        if resp.status_code != 200:
            update_transfer_task(task_id, status='error', message=f"远程返回{resp.status_code}: {resp.text[:200]}")
            return
        update_transfer_task(task_id, status='success', progress=100, message='上传完成')
    except Exception as e:
        update_transfer_task(task_id, status='error', message=str(e))


def upload_worker_dataset(task_id, target_url, safe_rel, file_path, token=None, overwrite=False):
    update_transfer_task(task_id, status='running', progress=0, message='准备上传数据集')
    try:
        file_size = os.path.getsize(file_path)
        headers = {
            'Content-Type': 'application/octet-stream',
            'X-Filename': os.path.basename(safe_rel)
        }
        if token:
            headers['X-Transfer-Token'] = token
        params = {'overwrite': '1' if overwrite else '0'}
        bytes_sent = 0

        def stream_file():
            nonlocal bytes_sent
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    bytes_sent += len(chunk)
                    if file_size:
                        progress = min(100, round(bytes_sent / file_size * 100, 2))
                    else:
                        progress = 100
                    update_transfer_task(task_id, status='running', progress=progress, message='上传数据集...')
                    yield chunk

        resp = requests.post(
            f"{target_url}/api/receive-dataset",
            params=params,
            data=stream_file(),
            headers=headers,
            timeout=600
        )
        if resp.status_code != 200:
            update_transfer_task(task_id, status='error', message=f"远程返回{resp.status_code}: {resp.text[:200]}")
            return
        update_transfer_task(task_id, status='success', progress=100, message='数据集上传完成')
    except Exception as e:
        update_transfer_task(task_id, status='error', message=str(e))


def pull_worker(task_id, source_url, safe_rel, dest_path, token=None, overwrite=False):
    update_transfer_task(task_id, status='running', progress=0, message='准备下载')
    ensure_out_dir()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and not overwrite:
        update_transfer_task(task_id, status='error', message='文件已存在且未开启覆盖')
        return
    headers = {}
    if token:
        headers['X-Transfer-Token'] = token
    params = {'filename': safe_rel}
    try:
        with requests.get(
            f"{source_url}/api/download-weight",
            headers=headers,
            params=params,
            stream=True,
            timeout=600
        ) as resp:
            if resp.status_code != 200:
                try:
                    detail = resp.json()
                except Exception:
                    detail = {'raw': resp.text[:200]}
                update_transfer_task(task_id, status='error', message=f"远程返回{resp.status_code}: {detail}")
                return
            total_size = int(resp.headers.get('Content-Length', 0)) if resp.headers.get('Content-Length') else 0
            bytes_written = 0
            with open(dest_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bytes_written += len(chunk)
                    if total_size:
                        progress = min(100, round(bytes_written / total_size * 100, 2))
                    else:
                        # 当无法获取总大小时，用写入量的近似值驱动进度，但保留到99%
                        current = get_transfer_task(task_id) or {}
                        progress = min(99, current.get('progress', 0) + 1)
                    update_transfer_task(task_id, status='running', progress=progress, message='下载中...')
            update_transfer_task(task_id, status='success', progress=100, message='下载完成')
    except Exception as e:
        update_transfer_task(task_id, status='error', message=str(e))


def derive_from_weight_prefix(filename):
    """根据文件名推测from_weight前缀"""
    base = os.path.splitext(os.path.basename(filename))[0]
    # 尝试去掉末尾的 _数字 或 _数字_moe
    parts = base.split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].lower() == 'moe':
        return '_'.join(parts[:-2])
    return base


def push_weight_to_callback(callback_url, file_path, token=None):
    if not callback_url:
        return False, '无回传地址'
    try:
        headers = {
            'Content-Type': 'application/octet-stream',
            'X-Filename': os.path.basename(file_path)
        }
        if token:
            headers['X-Transfer-Token'] = token
        with open(file_path, 'rb') as f:
            resp = requests.post(
                f"{callback_url}/api/receive-weight",
                params={'overwrite': '1'},
                headers=headers,
                data=iter(lambda: f.read(1024 * 1024), b''),
                timeout=600
            )
        if resp.status_code == 200:
            return True, '回传成功'
        return False, f"回传失败: {resp.status_code}"
    except Exception as e:
        return False, str(e)


def load_eval_model(params):
    """按eval_llm.py逻辑加载模型"""
    load_from = params.get('load_from', 'model')
    save_dir = params.get('save_dir', 'out')
    weight = params.get('weight', 'full_sft')
    lora_weight = params.get('lora_weight', 'None')
    hidden_size = int(params.get('hidden_size', 512))
    num_hidden_layers = int(params.get('num_hidden_layers', 8))
    use_moe = int(params.get('use_moe', 0))
    inference_rope_scaling = bool(params.get('inference_rope_scaling', False))
    device = params.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if 'model' in load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            use_moe=bool(use_moe),
            inference_rope_scaling=inference_rope_scaling
        ))
        moe_suffix = '_moe' if use_moe else ''
        ckp = os.path.abspath(os.path.join(PROJECT_ROOT, save_dir, f"{weight}_{hidden_size}{moe_suffix}.pth"))
        if not os.path.exists(ckp):
            raise FileNotFoundError(f"未找到权重文件: {ckp}")
        state = torch.load(ckp, map_location=device)
        model.load_state_dict(state, strict=True)
        if lora_weight and lora_weight != 'None':
            apply_lora(model)
            lora_path = os.path.abspath(os.path.join(PROJECT_ROOT, save_dir, 'lora', f"{lora_weight}_{hidden_size}.pth"))
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"未找到LoRA文件: {lora_path}")
            load_lora(model, lora_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    model.eval().to(device)
    return model, tokenizer, device


def run_eval_once(model, tokenizer, device, prompt, params, conversation=None):
    conversation = conversation or []
    conversation = conversation[-params['historys']:] if params['historys'] else []
    conversation.append({"role": "user", "content": prompt})

    templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
    if params['weight'] == 'reason':
        templates["enable_thinking"] = True
    inputs_text = tokenizer.apply_chat_template(**templates) if params['weight'] != 'pretrain' else (tokenizer.bos_token + prompt)
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(device)

    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=params['max_new_tokens'],
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=params['top_p'],
        temperature=params['temperature'],
        repetition_penalty=1.0
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": response})
    return response, conversation


@app.route('/api/eval/run', methods=['POST'])
def api_eval_run():
    data = request.get_json(silent=True) or {}
    try:
        params = {
            'load_from': data.get('load_from') or 'model',
            'save_dir': data.get('save_dir') or 'out',
            'weight': data.get('weight') or 'full_sft',
            'lora_weight': data.get('lora_weight') or 'None',
            'hidden_size': int(data.get('hidden_size') or 512),
            'num_hidden_layers': int(data.get('num_hidden_layers') or 8),
            'use_moe': int(data.get('use_moe') or 0),
            'inference_rope_scaling': bool(data.get('inference_rope_scaling') or False),
            'max_new_tokens': int(data.get('max_new_tokens') or 512),
            'temperature': float(data.get('temperature') or 0.85),
            'top_p': float(data.get('top_p') or 0.85),
            'historys': int(data.get('historys') or 0),
            'device': data.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
        }
    except Exception as e:
        return jsonify({'error': f'参数错误: {str(e)}'}), 400

    prompts = []
    if data.get('use_default_prompts'):
        prompts = [
            '你有什么特长？',
            '为什么天空是蓝色的',
            '请用Python写一个计算斐波那契数列的函数',
            '解释一下\"光合作用\"的基本过程',
            '如果明天下雨，我应该如何出门',
            '比较一下猫和狗作为宠物的优缺点',
            '解释什么是机器学习',
            '推荐一些中国的美食'
        ]
    else:
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'error': '请提供prompt或选择默认测试'}), 400
        prompts = [prompt]

    try:
        model, tokenizer, device = load_eval_model(params)
    except Exception as e:
        return jsonify({'error': f'模型加载失败: {str(e)}'}), 500

    outputs = []
    conversation = []
    try:
        for p in prompts:
            setup_seed(2026)
            resp, conversation = run_eval_once(model, tokenizer, device, p, params, conversation)
            outputs.append({'prompt': p, 'response': resp})
    except Exception as e:
        return jsonify({'error': f'推理失败: {str(e)}'}), 500

    return jsonify({'success': True, 'outputs': outputs})

def watch_lora_training(task_id, process_id, result_file, callback):
    """监控LoRA任务，完成后可选回传"""
    while True:
        info = training_processes.get(process_id)
        if not info:
            update_lora_task(task_id, status='error', message='找不到训练进程')
            return
        if not info.get('running'):
            if info.get('error'):
                update_lora_task(task_id, status='error', message='训练失败')
                return
            # 检查结果文件
            if not os.path.exists(result_file):
                update_lora_task(task_id, status='error', message='训练完成但未找到权重文件')
                return
            update_lora_task(task_id, status='success', message='训练完成', result_file=os.path.relpath(result_file, OUT_DIR))
            # 可选回传
            if callback and callback.get('url'):
                ok, msg = push_weight_to_callback(callback.get('url'), result_file, callback.get('token'))
                update_lora_task(task_id, callback_result=msg, callback_success=ok)
            return
        time.sleep(3)

@app.route('/logs/<process_id>')
def logs(process_id):
    # 直接从本地logfile目录读取日志文件
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    # 查找匹配的日志文件
    log_file = None
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if filename.endswith(f'{process_id}.log'):
                log_file = os.path.join(log_dir, filename)
                break
    
    if not log_file or not os.path.exists(log_file):
        return '日志文件不存在或已被删除'
    
    try:
        # 使用高效且健壮的方法读取文件的最后200行
        def read_last_n_lines(file_path, n=200):
            # 使用二进制模式读取文件，避免编码问题
            with open(file_path, 'rb') as f:
                # 获取文件大小
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                
                # 如果文件很小，直接读取整个文件
                if file_size < 1024 * 1024:  # 小于1MB的文件直接读取
                    f.seek(0)
                    content = f.read()
                    return process_content(content)
                
                # 对于大文件，使用缓冲读取末尾部分
                # 估计需要读取的字节数（假设每行平均100字节）
                buffer_size = n * 200  # 为了保险，读取更多字节
                
                # 定位到适当的位置
                position = max(0, file_size - buffer_size)
                f.seek(position)
                
                # 读取缓冲区内容
                buffer = f.read(file_size - position)
                
                # 处理缓冲区内容
                lines = process_content(buffer)
                
                # 确保我们获取到完整的行
                # 如果缓冲区不是从文件开头开始，第一个行可能不完整
                if position > 0:
                    # 跳过第一个可能不完整的行
                    if len(lines) > 1:
                        lines = lines[1:]
                    else:
                        # 如果只有一行且不在文件开头，可能需要读取更多
                        # 这里简单处理，直接读取整个文件（罕见情况）
                        f.seek(0)
                        content = f.read()
                        lines = process_content(content)
                
                # 返回最后n行
                return lines[-n:] if len(lines) > n else lines
        
        def process_content(content):
            # 尝试多种编码方式解码内容
            encodings = ['utf-8', 'latin-1', 'gbk', 'gb2312']
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    # 使用True参数保留换行符，确保行分隔符正确
                    return text.splitlines(True)
                except UnicodeDecodeError:
                    continue
            # 如果所有编码都失败，使用错误替换模式
            text = content.decode('utf-8', errors='replace')
            return text.splitlines(True)
        
        # 读取最后200行
        last_200_lines = read_last_n_lines(log_file, 200)
        
        # 确保返回的内容顺序正确，并且不包含空行
        return ''.join(last_200_lines)
    except Exception as e:
        return f'读取日志失败: {str(e)}'

@app.route('/logfiles')
def get_logfiles():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    logfiles = []
    # 获取所有进程ID用于关联
    process_pids = set(training_processes.keys())
    
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if filename.endswith('.log') and filename.startswith('train_'):
                file_path = os.path.join(log_dir, filename)
                try:
                    modified_time = os.path.getmtime(file_path)
                    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modified_time))
                    # 提取进程ID
                    pid = filename.split('.')[-2].split('_')[-1] if filename.endswith('.log') else None
                    logfiles.append({
                        'filename': filename,
                        'modified_time': formatted_time,
                        'size': os.path.getsize(file_path),
                        'process_id': pid,
                        'has_process': pid in process_pids
                    })
                except Exception as e:
                    continue
    # 按修改时间倒序排序，最新的在前面
    logfiles.sort(key=lambda x: x['modified_time'], reverse=True)
    return jsonify(logfiles)

@app.route('/logfile-content/<filename>')
def get_logfile_content(filename):
    # 安全检查：确保文件名不包含路径遍历字符
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径，train_web_ui.py在scripts目录下
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    log_file = os.path.join(log_dir, filename)
    
    try:
        # 使用二进制模式读取文件，可以更可靠地保留原始换行符
        with open(log_file, 'rb') as f:
            content_bytes = f.read()
        
        # 尝试多种编码方式解码，确保正确处理换行符
        encodings = ['utf-8', 'latin-1', 'gbk', 'gb2312']
        content = None
        
        for encoding in encodings:
            try:
                # 解码文件内容，保留原始换行符
                content = content_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，使用errors='replace'参数处理不可解码的字符
        if content is None:
            content = content_bytes.decode('utf-8', errors='replace')
        
        # 确保返回的内容正确保留所有换行符
        return content
    except FileNotFoundError:
        return jsonify({'error': 'Log file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete-logfile/<filename>', methods=['DELETE'])
def delete_logfile(filename):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建logfile目录的绝对路径
    log_dir = os.path.join(script_dir, '../logfile')
    log_dir = os.path.abspath(log_dir)
    
    # 安全检查：防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'success': False, 'message': '非法的文件名'})
    
    log_file = os.path.join(log_dir, filename)
    if os.path.exists(log_file) and os.path.isfile(log_file):
        try:
            os.remove(log_file)
            return jsonify({'success': True, 'message': '日志文件删除成功'})
        except Exception as e:
            print(f"删除日志文件失败: {str(e)}")
            return jsonify({'success': False, 'message': f'删除失败: {str(e)}'})
    return jsonify({'success': False, 'message': '日志文件不存在'})


@app.route('/stop/<process_id>', methods=['POST'])
def stop(process_id):
    if process_id in training_processes and training_processes[process_id]['running']:
        process = training_processes[process_id]['process']
        # 在Windows上使用terminate，在Unix上尝试优雅终止
        try:
            process.terminate()
            # 等待进程结束
            process.wait(timeout=5)
            # 标记为手动停止
            training_processes[process_id]['running'] = False
            training_processes[process_id]['manually_stopped'] = True
        except subprocess.TimeoutExpired:
            # 如果超时，强制杀死
            process.kill()
            # 标记为手动停止
            training_processes[process_id]['running'] = False
            training_processes[process_id]['manually_stopped'] = True
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/delete/<process_id>', methods=['POST'])
def delete(process_id):
    if process_id in training_processes:
        # 确保进程已经停止
        if training_processes[process_id]['running']:
            # 如果进程还在运行，先停止它
            try:
                process = training_processes[process_id]['process']
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            except Exception as e:
                print(f"停止进程失败: {str(e)}")
        
        # 从进程字典中删除
        del training_processes[process_id]
        
        # 可选：删除对应的日志文件
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(script_dir, '../logfile')
            log_dir = os.path.abspath(log_dir)
            
            if os.path.exists(log_dir):
                for filename in os.listdir(log_dir):
                    if filename.endswith(f'{process_id}.log'):
                        os.remove(os.path.join(log_dir, filename))
        except Exception as e:
            print(f"删除日志文件失败: {str(e)}")
        
        return jsonify({'success': True})
    return jsonify({'success': False})

def find_available_port(preferred=None, allowed_ports=None):
    """在限定端口中查找可用端口（autodl 仅开放 6006/6008）"""
    allowed = allowed_ports or ALLOWED_PORTS
    ordered = []
    if preferred and preferred in allowed:
        ordered.append(preferred)
    ordered.extend([p for p in allowed if p not in ordered])
    for port in ordered:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        if result != 0:
            return port
    return None

def save_processes_info():
    """保存训练进程信息到文件"""
    try:
        # 创建一个不包含进程对象的可序列化版本
        serializable_processes = {}
        for pid, info in training_processes.items():
            serializable_processes[pid] = {
                'pid': info.get('pid', info.get('process').pid) if isinstance(info.get('process'), subprocess.Popen) else info.get('pid'),
                'train_type': info['train_type'],
                'log_file': info['log_file'],
                'start_time': info['start_time'],
                'running': info['running'],
                'error': info.get('error', False),
                'manually_stopped': info.get('manually_stopped', False),
                'train_monitor': info.get('train_monitor', 'none'),  # 保存train_monitor
                'swanlab_url': info.get('swanlab_url'),
                'client_id': info.get('client_id')
            }
        
        with open(PROCESSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_processes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存进程信息失败: {str(e)}")

def load_processes_info():
    """从文件加载训练进程信息"""
    global training_processes
    try:
        if os.path.exists(PROCESSES_FILE):
            with open(PROCESSES_FILE, 'r', encoding='utf-8') as f:
                loaded_processes = json.load(f)
            
            # 检查每个进程是否还在运行
            for pid, info in loaded_processes.items():
                # 确保所有需要的字段都存在
                if 'swanlab_url' not in info:
                    info['swanlab_url'] = None
                if 'manually_stopped' not in info:
                    info['manually_stopped'] = False
                if 'error' not in info:
                    info['error'] = False
                if 'train_monitor' not in info:
                    info['train_monitor'] = 'none'
                if 'client_id' not in info:
                    info['client_id'] = None
                
                if info['running']:
                    try:
                        # 检查进程是否还在运行
                        proc = psutil.Process(info['pid'])
                        if proc.is_running() and proc.status() != 'zombie':
                            # 进程仍在运行，恢复信息
                            training_processes[pid] = info
                        else:
                            # 进程已停止
                            info['running'] = False
                            # 如果进程未被明确标记为完成或出错，则默认为手动停止
                            if not info['error']:
                                info['manually_stopped'] = True
                            training_processes[pid] = info
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # 进程不存在或无权限访问
                        info['running'] = False
                        # 如果进程未被明确标记为完成或出错，则默认为手动停止
                        if not info['error']:
                            info['manually_stopped'] = True
                        training_processes[pid] = info
                else:
                    # 进程已停止，直接恢复
                    training_processes[pid] = info
    except Exception as e:
        print(f"加载进程信息失败: {str(e)}")

def handle_exit(signum, frame):
    """处理程序退出信号，保存进程信息"""
    print("正在保存进程信息...  save at 'trainer_web/training_processes.json'...")
    save_processes_info()
    # 删除PID文件
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except:
            pass
    sys.exit(0)

# 注册退出处理器
signal.signal(signal.SIGINT, handle_exit)  # Ctrl+C
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, handle_exit)  # 终止信号

# 注册程序退出时的处理函数
atexit.register(save_processes_info)

if __name__ == '__main__':
    # 加载已保存的进程信息
    load_processes_info()
    
    # 创建PID文件，用于标识web进程
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    # autodl 仅开放 6006/6008；优先使用环境变量 FLASK_PORT/PORT
    preferred_env_port = os.environ.get('FLASK_PORT') or os.environ.get('PORT')
    preferred_env_port = int(preferred_env_port) if preferred_env_port and str(preferred_env_port).isdigit() else None
    port = find_available_port(preferred=preferred_env_port)
    if port is not None:
        print(f"启动Flask服务器在 http://0.0.0.0:{port}")
        print(f"使用nohup启动可保持服务持续运行: nohup python -u scripts/train_web_ui.py &")
        # 使用0.0.0.0作为host以兼容VSCode的端口转发功能
        app.run(host='0.0.0.0', port=port, debug=False)  # 生产环境关闭debug
    else:
        print(f"无法找到可用端口，请检查 {ALLOWED_PORTS} 是否被占用")
        # 删除PID文件
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except:
                pass
        sys.exit(1)
# Registration endpoint removed - allow anonymous training
