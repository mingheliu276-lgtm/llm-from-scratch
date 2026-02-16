#!/bin/bash
# 云服务器训练脚本（Linux / AutoDL）
# 用法: 先 cd 到项目根目录，再 bash scripts/train_cloud.sh
# 若已创建 venv，建议先 source venv/bin/activate，脚本会优先用 venv 的 Python

# 进入脚本所在目录的上级（项目根）
cd "$(dirname "$0")/.." || exit 1

# 优先用当前目录下的 venv（保证用到装好 torch+cuda 的环境）
if [ -x "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
    echo "使用 venv: $PYTHON_CMD"
else
    PYTHON_CMD="python"
fi

# 设置参数（当前按 V100 32GB 调优；其他显卡见 云GPU训练指南.md）
D_MODEL=512
NUM_LAYERS=6
NUM_HEADS=8
BATCH_SIZE=24
NUM_EPOCHS=10
MAX_SEQ_LEN=512

# AutoDL 数据盘路径（checkpoint 保存到数据盘，关机不丢）
if [ -d /root/autodl-tmp ]; then
    SAVE_DIR="/root/autodl-tmp/checkpoints"
else
    SAVE_DIR="checkpoints"
fi

# 国内云服务器无法直连 Hugging Face，用镜像下载数据集和 tokenizer
export HF_ENDPOINT=https://hf-mirror.com

# 后台运行训练，日志输出到 train.log
nohup "$PYTHON_CMD" scripts/train_with_real_data.py \
    --dataset wikitext2 \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --use_amp \
    --max_seq_len $MAX_SEQ_LEN \
    --save_dir "$SAVE_DIR" \
    > train.log 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "查看日志: tail -f train.log"
echo "查看GPU: watch -n 1 nvidia-smi"
