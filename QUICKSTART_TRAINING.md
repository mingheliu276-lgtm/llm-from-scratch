# 🚀 快速开始 - 从零训练大模型

## 第一步：检查环境

```powershell
# 1. 检查CUDA是否可用
python scripts/cuda_basics.py
```

如果看到 "CUDA可用: True"，说明环境配置正确！

## 第二步：测试模型组件

```powershell
# 测试Attention机制
python src/models/attention.py

# 测试Transformer模型
python src/models/transformer.py

# 测试训练器
python src/training/trainer.py
```

## 第三步：开始第一次训练

```powershell
# 小模型配置（适合RTX 5060，约1M参数）
python scripts/train.py \
    --vocab_size 5000 \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 2 \
    --d_ff 1024 \
    --max_seq_len 128 \
    --batch_size 4 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --use_amp \
    --gradient_accumulation_steps 2
```

## 监控GPU使用

在另一个终端运行：
```powershell
# Windows PowerShell
while ($true) { nvidia-smi; Start-Sleep -Seconds 2 }
```

## 理解代码结构

### 1. `src/models/attention.py`
- **MultiHeadAttention**: 多头注意力机制
- **PositionalEncoding**: 位置编码

### 2. `src/models/transformer.py`
- **SimpleGPT**: 完整的Transformer模型
- 包含Embedding、Encoder、输出层

### 3. `src/training/trainer.py`
- **Trainer**: 训练循环
- 支持混合精度训练（FP16）
- 支持梯度累积
- 自动保存检查点

### 4. `scripts/train.py`
- 训练入口脚本
- 可配置所有超参数

## 下一步学习

1. **阅读代码**: 理解每个组件的实现
2. **修改超参数**: 尝试不同的配置
3. **添加功能**: 
   - 实现真正的Causal Mask（GPT风格）
   - 添加学习率调度器
   - 实现LoRA微调
4. **使用真实数据**: 加载WikiText-2等数据集

## 常见问题

**Q: 训练很慢怎么办？**  
A: 
- 确保使用了 `--use_amp`（混合精度）
- 减小 `--max_seq_len`
- 减小 `--batch_size` 但增加 `--gradient_accumulation_steps`

**Q: 显存不足？**  
A:
- 减小 `--d_model` 和 `--num_layers`
- 减小 `--batch_size`
- 使用 `--gradient_accumulation_steps` 模拟更大的batch

**Q: 如何保存和加载模型？**  
A: 训练器会自动保存检查点到 `checkpoints/` 目录

## 推荐学习顺序

1. ✅ 运行 `cuda_basics.py` 熟悉GPU
2. ✅ 阅读 `attention.py` 理解注意力机制
3. ✅ 阅读 `transformer.py` 理解完整架构
4. ✅ 运行小规模训练（1M参数）
5. ✅ 逐步增加模型大小
6. ✅ 使用真实数据集训练

---

**记住**: 从简单开始，逐步增加复杂度！🎯
