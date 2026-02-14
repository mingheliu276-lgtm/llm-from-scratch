# 从零训练大模型 - LLM From Scratch

你好！这是一个从零开始训练大模型的完整项目，专为EE专业学生设计，目标是掌握Transformer架构、CUDA编程和大模型训练流程。

## 🎯 项目目标

- **理解Transformer架构** - 掌握注意力机制、位置编码等核心概念
- **掌握CUDA编程** - 熟悉GPU并行计算、内存管理（使用RTX 5060）
- **实现训练流程** - 从数据加载到模型训练完整流程
- **性能优化** - 混合精度训练、梯度累积、分布式训练

## 📚 学习路径

查看 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** 了解完整的4周学习计划。

## 🚀 快速开始

### 1. 检查CUDA环境

```powershell
cd d:\llm-from-scratch
python scripts/cuda_basics.py
```

CUDA 与 GPU 用法详解见 **[CUDA基础语法教程.md](CUDA基础语法教程.md)**。

### 2. 安装依赖

```powershell
# 创建虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1

# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 测试模型组件

```powershell
# 测试Attention机制
python src/models/attention.py

# 测试Transformer模型
python src/models/transformer.py

# 测试训练器
python src/training/trainer.py
```

### 4. 开始第一次训练

#### 选项A：使用虚拟数据集（快速测试训练流程）

```powershell
# 小模型配置（适合RTX 5060，约7M参数）
python scripts/train.py \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 2 \
    --batch_size 4 \
    --num_epochs 5 \
    --use_amp
```

**注意**: 虚拟数据集训练的模型无法回答真实问题，仅用于测试训练流程。  
损失约 9.2、各项指标含义见 **[理解训练输出.md](理解训练输出.md)**。

#### 选项B：使用真实数据集（推荐，能回答问题）

```powershell
# 使用WikiText-2数据集训练
python scripts/train_with_real_data.py \
    --dataset wikitext2 \
    --d_model 256 \
    --num_layers 2 \
    --num_heads 4 \
    --batch_size 4 \
    --use_amp
```

**首次运行需要下载数据集**（约4MB），会自动下载。

### 5. 使用训练好的模型

```powershell
# 生成文本
python scripts/inference.py --prompt "Hello, how are you?"

# 对话模式
python scripts/chat.py
```

## 📁 项目结构

```
llm-from-scratch/
├── README.md                 # 项目说明
├── TRAINING_GUIDE.md         # 完整学习路径（4周计划）
├── QUICKSTART_TRAINING.md    # 快速入门指南
├── requirements.txt          # Python依赖
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py      # Multi-Head Attention实现
│   │   └── transformer.py    # Transformer模型
│   └── training/
│       ├── __init__.py
│       └── trainer.py        # 训练循环
└── scripts/
    ├── train.py              # 训练脚本
    └── cuda_basics.py        # CUDA基础练习
```

## 📊 数据集说明

### 当前使用的数据集

**虚拟数据集**（默认）:
- 随机token序列，仅用于测试训练流程
- ❌ 无法学习语言知识
- ❌ 训练出的模型无法回答真实问题

**真实数据集**（推荐）:
- WikiText-2: 维基百科文章，适合学习
- 查看 [DATASET_INFO.md](DATASET_INFO.md) 了解更多

### 模型能做什么？

✅ **文本生成**: 给定提示，继续生成文本  
✅ **文本补全**: 自动补全句子  
⚠️ **问答**: 需要真实数据集训练才能有意义的回答  
⚠️ **对话**: 小模型能力有限，建议使用更大模型或微调

## 💡 推荐学习顺序

1. ✅ 运行 `scripts/cuda_basics.py` 熟悉GPU
2. ✅ 阅读 `src/models/attention.py` 理解注意力机制
3. ✅ 阅读 `src/models/transformer.py` 理解完整架构（详解见 **[理解Transformer架构.md](理解Transformer架构.md)**）
4. ✅ 使用虚拟数据集测试训练流程
5. ✅ **使用真实数据集训练**（WikiText-2）
6. ✅ 使用 `scripts/inference.py` 和 `scripts/chat.py` 测试模型
7. ✅ 逐步增加模型大小和数据集规模

## 📖 推荐资源

- **论文**: Attention Is All You Need (Transformer原始论文)
- **教程**: Jay Alammar的The Illustrated Transformer
- **代码**: nanoGPT (Karpathy) - 最小GPT实现

## 🎓 下一步

查看 **[QUICKSTART_TRAINING.md](QUICKSTART_TRAINING.md)** 开始你的第一次训练！

## ☁️ 云GPU训练

想在租用的服务器上训练更大模型？查看 **[云GPU训练指南.md](云GPU训练指南.md)** 了解如何在 AutoDL、阿里云等平台运行训练。

---

**记住**: 从简单开始，逐步增加复杂度！🚀
