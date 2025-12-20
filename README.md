# MiniMind 训练与 Web UI 运行指南

## 环境准备
- Python 3.12+（建议使用 uv）
- GPU 服务器（autodl 平台），开放端口：6006 或 6008
- 本地或远端已同步的代码仓库

## 安装 uv 与依赖
1. 安装 uv（Linux/macOS）：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Windows 请参考官方文档。
2. 安装依赖（会使用 pyproject.toml）：
    ```bash
    uv sync
    ```
    或者
    ```bash
    uv run
    ```
    uv 工具会自动创建并激活虚拟环境。
## 运行 Web UI（autodl）
1. 进入仓库目录：
   ```bash
   cd /path/to/cloud
   ```
   切换到 `dev` 分支
    ```bash
    git checkout dev
    git pull
    ```
2. 可选：设置端口（autodl 默认开放 6006/6008，默认使用 6006）：
   ```bash
   export FLASK_PORT=6006
   ```
3. 可选：设置传输鉴权令牌（跨服传输/LoRA 联训回传）：
   ```bash
   export TRANSFER_TOKEN=your-secret
   ```
4. 启动 Web UI：
   ```bash
   bash trainer_web/start_web_ui.sh
   ```
   日志位于 `logfile/web_ui_*.log`。

## autodl 平台获取外网访问链接
1. 在 autodl 控制台，选择对应实例，进入「自定义服务」。
2. 将 6006（或 6008）映射到公网，获得访问 URL。
3. 使用浏览器打开该 URL，即可访问训练/评测 Web 界面。

## 常见路径
- 训练输出：`out/`（LoRA 权重位于 `out/lora/`）
- 数据集：`dataset/`
- 日志：`logfile/`

## 停止服务
```bash
bash trainer_web/stop_web_ui.sh
```
