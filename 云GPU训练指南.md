# 云GPU训练指南：在租用的服务器上跑大模型

你想在云GPU上训练是对的——**个人显卡确实很难训出媲美开源大模型的效果**。这份指南帮你把项目搬到云服务器上，用更强的算力训练。

---

## 一、云GPU平台推荐（国内可用）

### 1. AutoDL（推荐，性价比高）

- **网址**：https://www.autodl.com/
- **优势**：
  - 按小时计费，用多少付多少
  - RTX 4090 / A100 等高端卡可选
  - 预装 PyTorch、CUDA，开箱即用
  - 支持 Jupyter、SSH、VSCode Remote
  - 数据盘持久化（关机不丢数据）
- **价格参考**：RTX 4090 约 2-3 元/小时，A100 约 8-10 元/小时
- **适合**：个人学习、小规模实验

### 2. 阿里云 / 腾讯云 / 华为云

- **优势**：稳定、企业级服务
- **劣势**：价格较高，配置相对复杂
- **适合**：企业项目、长期使用

### 3. 其他平台

- **Featurize**：https://featurize.cn/
- **恒源云**：https://www.gpushare.com/
- **矩池云**：https://www.matpool.com/

---

## 二、准备工作：本地代码整理

### 2.1 确保代码可移植

你的项目已经比较干净了，但建议：

1. **检查硬编码路径**：
   ```python
   # ❌ 避免硬编码
   save_dir = Path("D:/llm-from-scratch/checkpoints")
   
   # ✅ 使用相对路径
   save_dir = Path("checkpoints")
   ```

2. **确保 requirements.txt 完整**：
   ```bash
   # 在项目根目录运行，生成依赖列表
   pip freeze > requirements.txt
   ```
   或者手动维护：
   ```
   torch>=2.0.0
   transformers>=4.30.0
   datasets>=2.14.0
   ```

### 2.2 准备上传的文件

**必须上传**：
- `src/` 目录（所有模型和训练代码）
- `scripts/` 目录（训练/推理脚本）
- `requirements.txt`（依赖列表）
- `README.md`（可选，方便查看）

**不需要上传**：
- `venv/`（虚拟环境，在服务器上重建）
- `checkpoints/`（模型文件，在服务器上训练生成）
- `__pycache__/`（Python 缓存）

---

## 三、AutoDL 使用流程（详细步骤）

### 3.1 注册并创建实例

1. **注册账号**：https://www.autodl.com/
2. **充值**：建议先充 50-100 元测试
3. **创建实例**：
   - 选择 GPU：RTX 4090（24GB）或 A100（40GB/80GB）
   - 选择镜像：**PyTorch 2.x + CUDA 11.8**（或更新版本）
   - 数据盘：建议 50GB+（存放数据集和模型）
   - 开机后自动进入 Jupyter 界面

### 3.2 上传代码

**方法1：Jupyter 上传（最简单）**

1. 在 Jupyter 界面点击「上传」按钮
2. 选择项目文件夹（压缩成 zip 后上传）
3. 在 Jupyter 中解压：
   ```bash
   !unzip llm-from-scratch.zip
   ```

**方法2：Git 克隆（推荐）**

如果你把代码放到 GitHub/Gitee：

```bash
# 在服务器上
git clone https://github.com/your-username/llm-from-scratch.git
cd llm-from-scratch
```

**方法3：SSH + scp（适合大文件）**

```bash
# 在本地电脑
scp -r d:/llm-from-scratch root@your-server-ip:/root/
```

### 3.3 安装依赖

```bash
# SSH 进入服务器，或 Jupyter 的 Terminal
cd llm-from-scratch

# 创建虚拟环境（可选，但推荐）
python -m venv venv
source venv/bin/activate  # Linux，Windows 用 venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 如果没有 requirements.txt，手动安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
```

### 3.4 验证环境

```bash
# 检查 GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 测试训练流程（小规模）
python scripts/train.py --d_model 128 --num_layers 1 --batch_size 2 --num_epochs 1
```

---

## 四、开始训练（云服务器上）

### 4.1 使用真实数据训练

```bash
# 进入项目目录
cd llm-from-scratch

# 激活虚拟环境（如果用了）
source venv/bin/activate

# 开始训练（根据你的 GPU 显存调整 batch_size）
python scripts/train_with_real_data.py \
    --dataset wikitext2 \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 16 \
    --num_epochs 10 \
    --use_amp \
    --max_seq_len 512
```

### 4.2 后台运行（避免 SSH 断开导致训练中断）

**方法1：使用 nohup**

```bash
nohup python scripts/train_with_real_data.py \
    --dataset wikitext2 \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 16 \
    --num_epochs 10 \
    --use_amp \
    > train.log 2>&1 &

# 查看日志
tail -f train.log
```

**方法2：使用 screen 或 tmux**

```bash
# 安装 screen
apt-get install screen  # 或 yum install screen

# 创建新会话
screen -S training

# 在会话中运行训练
python scripts/train_with_real_data.py ...

# 按 Ctrl+A 然后 D 退出会话（训练继续）
# 重新连接：screen -r training
```

**方法3：AutoDL 的「无卡模式」**

- AutoDL 支持「无卡模式」：训练时 GPU 计费，不训练时只收存储费
- 训练脚本会自动保存 checkpoint，下次开机继续训练

### 4.3 监控训练进度

```bash
# 查看 GPU 使用情况
nvidia-smi

# 查看训练日志
tail -f train.log

# 查看已保存的模型
ls -lh checkpoints/
```

---

## 五、下载训练好的模型

### 5.1 从服务器下载到本地

**方法1：Jupyter 下载**

- 在 Jupyter 界面找到 `checkpoints/best_model.pt`
- 右键 → Download

**方法2：scp 下载**

```bash
# 在本地电脑运行
scp root@your-server-ip:/root/llm-from-scratch/checkpoints/best_model.pt ./
```

**方法3：AutoDL 数据盘**

- AutoDL 的数据盘可以「下载」，直接下载整个 `checkpoints/` 文件夹

### 5.2 在本地使用模型

```bash
# 把下载的 best_model.pt 放到本地项目的 checkpoints/ 目录
python scripts/inference.py --prompt "Hello"
python scripts/chat.py
```

---

## 六、成本优化建议

### 6.1 选择合适的 GPU

| GPU | 显存 | 价格/小时 | 适合模型大小 |
|-----|------|-----------|--------------|
| RTX 3090 | 24GB | ~2元 | d_model≤768, 6层 |
| RTX 4090 | 24GB | ~2.5元 | d_model≤1024, 12层 |
| A100 40GB | 40GB | ~8元 | d_model≤2048, 24层+ |
| A100 80GB | 80GB | ~12元 | 超大模型 |

**建议**：先用 RTX 4090 测试，确认流程无误后再上 A100。

### 6.2 训练策略

1. **先用小模型验证**：
   ```bash
   # 快速验证（1-2小时）
   python scripts/train_with_real_data.py --d_model 256 --num_layers 2 --num_epochs 1
   ```

2. **逐步放大**：
   - 确认无误后，再加大 `--d_model`、`--num_layers`
   - 根据显存调整 `--batch_size`

3. **保存检查点**：
   - 每几个 epoch 保存一次，避免意外中断丢失进度
   - 训练脚本已自动保存 `best_model.pt`

### 6.3 节省成本

- **使用混合精度**：`--use_amp` 可省约 50% 显存，速度提升 1.5-2x
- **梯度累积**：小 batch + 梯度累积 = 大 batch 效果，但显存占用小
- **及时关机**：训练完成后及时关机，只收存储费（AutoDL 约 0.1元/天）

---

## 七、常见问题

### Q1: SSH 断开后训练中断？

**A**: 使用 `nohup`、`screen` 或 `tmux` 后台运行（见 4.2 节）。

### Q2: 数据下载慢？

**A**: 
- WikiText-2 很小（~5MB），一般很快
- 如果慢，可以在本地下载后上传到服务器
- 或使用镜像源（AutoDL 通常已配置）

### Q3: 显存不足（OOM）？

**A**: 
```bash
# 减小 batch_size
--batch_size 8  # 或更小

# 启用混合精度
--use_amp

# 减小模型
--d_model 384 --num_layers 4

# 梯度累积（模拟大 batch）
# 修改 trainer.py 的 gradient_accumulation_steps=4
```

### Q4: 训练速度慢？

**A**: 
- 检查 `nvidia-smi`，确认 GPU 利用率 >80%
- 启用 `--use_amp`（混合精度）
- 增大 `--batch_size`（在显存允许范围内）
- 检查数据加载：`num_workers=4`（Linux）或 `num_workers=0`（Windows）

### Q5: 如何续训（从 checkpoint 继续）？

**A**: 需要修改训练脚本，添加 `--resume` 参数：

```python
# 在 train_with_real_data.py 的 main() 中添加
if args.resume:
    trainer.load_checkpoint(Path(args.resume))
    trainer.train(num_epochs=args.num_epochs)
```

---

## 八、完整示例：AutoDL 上训练 6 层模型

```bash
# 1. SSH 登录服务器
ssh root@your-instance-id

# 2. 进入项目目录
cd llm-from-scratch

# 3. 激活环境
source venv/bin/activate

# 4. 后台训练（6层，d_model=768，适合 RTX 4090）
nohup python scripts/train_with_real_data.py \
    --dataset wikitext2 \
    --d_model 768 \
    --num_layers 6 \
    --num_heads 12 \
    --batch_size 8 \
    --num_epochs 20 \
    --use_amp \
    --max_seq_len 512 \
    > train_6layer.log 2>&1 &

# 5. 查看进度
tail -f train_6layer.log

# 6. 训练完成后下载模型
# 在 Jupyter 或 scp 下载 checkpoints/best_model.pt
```

**预计时间**：RTX 4090 上约 10-20 小时（取决于数据量和 epoch 数）

**预计成本**：约 25-50 元

---

## 九、进阶：多GPU训练（可选）

如果你的预算充足，可以租多张 GPU 做分布式训练：

```python
# 需要修改 trainer.py，使用 torch.nn.DataParallel 或 DistributedDataParallel
# 这里不展开，需要时再详细说明
```

---

## 十、总结

**推荐流程**：

1. **本地准备**：整理代码，确保可移植
2. **选择平台**：AutoDL（性价比高）或其他
3. **上传代码**：Git 或直接上传
4. **安装依赖**：pip install
5. **小规模测试**：确认环境正常
6. **正式训练**：后台运行，监控日志
7. **下载模型**：训练完成后下载到本地使用

**成本参考**：
- **小模型**（2-4层，d_model≤512）：约 5-10 元
- **中等模型**（6-12层，d_model≤1024）：约 20-50 元
- **大模型**（24层+，d_model≥2048）：约 100-500 元+

**记住**：先用小模型验证流程，确认无误后再投入更多资源训练大模型！

祝你训练顺利～ 🚀
