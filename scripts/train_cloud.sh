#!/bin/bash
# 云服务器训练脚本（Linux）
# 用法: bash scripts/train_cloud.sh

# 设置参数（根据你的GPU显存调整）
D_MODEL=512
NUM_LAYERS=6
NUM_HEADS=8
BATCH_SIZE=16
NUM_EPOCHS=10
MAX_SEQ_LEN=512

# 后台运行训练，日志输出到 train.log
nohup python scripts/train_with_real_data.py \
    --dataset wikitext2 \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --use_amp \
    --max_seq_len $MAX_SEQ_LEN \
    > train.log 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "查看日志: tail -f train.log"
echo "查看GPU: watch -n 1 nvidia-smi"
