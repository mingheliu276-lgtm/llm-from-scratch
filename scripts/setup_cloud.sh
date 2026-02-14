#!/bin/bash
# 云服务器环境配置脚本
# 用法: bash scripts/setup_cloud.sh

echo "开始配置云服务器环境..."

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装 PyTorch（CUDA 11.8）
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
echo "安装其他依赖..."
pip install transformers datasets

# 验证 GPU
echo "验证 GPU 环境..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "环境配置完成！"
echo "激活环境: source venv/bin/activate"
