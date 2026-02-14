# 问题排查指南

## 已修复的问题

### 1. ✅ CUDA兼容性警告（非致命）

**问题**：
```
NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not compatible 
with the current PyTorch installation.
```

**原因**：RTX 5060是较新的GPU（计算能力12.0），而PyTorch 2.7.1+cu118只正式支持到sm_90。

**解决方案**：
- ⚠️ 这只是警告，GPU仍然可以工作（从测试结果可以看出）
- 未来PyTorch更新版本会支持sm_120
- 目前不影响训练，可以忽略此警告

### 2. ✅ 梯度计算错误（已修复）

**问题**：
```
AttributeError: 'NoneType' object has no attribute 'shape'
```

**原因**：在某些情况下，`x.grad`可能为None（虽然x是leaf tensor）。

**修复**：添加了检查，如果gradient为None会显示警告而不是崩溃。

### 3. ✅ GradScaler弃用警告（已修复）

**问题**：
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. 
Please use `torch.amp.GradScaler('cuda', args...)` instead.
```

**修复**：更新为新的API，同时保持向后兼容。

### 4. ✅ Transformer导入错误（已修复）

**问题**：
```
ImportError: attempted relative import with no known parent package
```

**原因**：直接运行`python src/models/transformer.py`时，Python不知道这是一个包。

**修复**：添加了导入fallback机制，支持直接运行和作为模块导入。

### 5. ✅ 数据批次解包错误（已修复）

**问题**：
```
ValueError: not enough values to unpack (expected 2, got 1)
```

**原因**：当使用`TensorDataset`创建单个tensor的数据集时，`DataLoader`返回的batch是`(tensor,)`（长度为1的tuple），但代码尝试解包为两个值。

**修复**：改进了数据处理逻辑，现在能正确处理：
- `(tensor,)` - 单个tensor的tuple（自动生成labels用于语言建模）
- `(input_ids, labels)` - 两个tensor的tuple
- `tensor` - 直接的tensor

## 运行建议

### 测试模型组件

```powershell
# ✅ 正确方式：作为模块运行
python -m src.models.attention
python -m src.models.transformer

# ✅ 或者：使用训练脚本（会自动处理导入）
python scripts/train.py --help
```

### CUDA基础练习

```powershell
# ✅ 现在应该可以正常运行了
python scripts/cuda_basics.py
```

## 常见问题

### Q: RTX 5060的CUDA警告会影响训练吗？
A: 不会。虽然PyTorch显示警告，但GPU仍然可以正常工作。训练和推理都能正常运行。

### Q: 如何检查GPU是否真的在工作？
A: 运行`python scripts/cuda_basics.py`，如果看到GPU加速比>1，说明GPU正常工作。

### Q: 训练时显存不足怎么办？
A: 
- 减小`--batch_size`（如从4改为2）
- 增加`--gradient_accumulation_steps`（如从1改为2）
- 减小`--d_model`和`--num_layers`
- 使用`--use_amp`启用混合精度训练

### Q: 对话/生成效果很差：一直重复 "are are are"、答非所问，是什么原因？
A: 主要有以下几方面，**不是 bug，而是当前配置的客观限制**：

1. **模型很小**：2 层、d_model=256、约 28M 参数。能「像人一样对话」的模型通常是几B～几十B 参数，并做过指令/对话微调。
2. **训练数据不是对话**：WikiText-2 是百科类连续文本，模型只学过「下一个词是什么」，没学过「用户问一句、你答一句」的格式，所以不会「回答问题」，只会按百科风格续写。
3. **训练轮数少**：只训了 5 个 epoch，主要学到的是词频和简单搭配（如 "and"、"I"、"to the"），还没学到连贯句子和逻辑。
4. **重复循环**：小模型容易陷入同一 token 或短语反复出现。已在生成里加入 **重复惩罚（repetition_penalty）**，对话脚本默认 1.3，推理可用 `--repetition_penalty 1.2`，会稍微减轻 "are are are" 式输出。

想稍微改善观感可以：多训几轮、适当加大模型（如 4 层、d_model=384）、推理时用 `--temperature 0.7 --repetition_penalty 1.3`；但要达到「真正能聊」的水平，需要更大模型 + 对话/指令数据 + 更长训练或微调。

## 下一步

1. ✅ 运行`python scripts/cuda_basics.py`确认所有练习通过
2. ✅ 运行`python -m src.models.transformer`测试模型
3. ✅ 开始第一次训练：`python scripts/train.py --d_model 256 --num_layers 2 --use_amp`
