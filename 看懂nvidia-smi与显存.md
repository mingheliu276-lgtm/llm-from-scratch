# 看懂 nvidia-smi：GPU 在怎样工作

你贴的 nvidia-smi 输出可以这样读。

---

## 一、你的输出在说什么

### 1. 上半部分：当前有哪些进程在用 GPU

```
|    0   N/A  N/A           35372      C   ...python.exe      N/A      |
```

- **GPU 0**：你只有一张卡，编号 0。
- **PID 35372**：占用 GPU 的进程 ID（就是你的训练 Python）。
- **Type C**：C = Compute（计算），说明是在跑 CUDA 计算，不是只用来显示画面。
- **GPU Memory Usage 列显示 N/A**：在 WDDM 驱动下，有时进程级显存不单独显示，看下面「整卡显存」更准。

### 2. 下半部分：整张卡的状态

| 项 | 你的值 | 含义 |
|----|--------|------|
| **Driver / CUDA** | 576.88 / 12.9 | 驱动和 CUDA 版本。 |
| **GPU 名称** | NVIDIA GeForce RTX 5060 ... | 显卡型号。 |
| **Memory-Usage** | **2187 MiB / 8151 MiB** | **当前显存用了约 2.2 GB，总共约 8 GB**。 |
| **GPU-Util** | **87%** | **GPU 计算利用率 87%**：算力在用，没有闲着。 |
| **Temp** | 76°C | 核心温度，正常范围。 |
| **Pwr: 67W / 66W** | 功耗接近标定上限，说明 GPU 在满负荷附近跑。 |

结论：**GPU 确实在工作（87% 利用率、功耗拉满），只是当前任务只占用了约 2.2 GB 显存，所以看起来「显存占用不够多」。**

---

## 二、当前训练配置（你用的命令）

你运行的命令是：

```text
python scripts/train_with_real_data.py --dataset wikitext2 --d_model 256 --num_layers 2 --num_heads 4 --batch_size 4 --use_amp
```

脚本里默认/未改动的关键参数是：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| **batch_size** | **4** | 每步 4 个样本，显存占用小。 |
| **num_epochs** | 5（默认） | 共训练 5 个 epoch。 |
| **d_model** | 256 | 模型宽度。 |
| **num_layers** | 2 | 只有 2 层，模型不大。 |
| **max_seq_len** | 128（默认） | 序列长度 128。 |
| **use_amp** | 是 | 混合精度，进一步省显存。 |

模型大约 **28M 参数**（vocab 50257），batch=4、seq_len=128，再加上 AMP，**用 2～2.5 GB 显存是正常的**。8 GB 卡上还有很多余量。

---

## 三、想多占显存、提吞吐可以怎么调

在**不爆显存**的前提下，想提高显存利用率和训练速度，可以适当加大 batch、或稍微加大模型/序列长度。例如（在 8 GB 显存下先保守试）：

- **只加大 batch**（最直接、显存涨得最明显）  
  ```bash
  python scripts/train_with_real_data.py --dataset wikitext2 --d_model 256 --num_layers 2 --num_heads 4 --batch_size 16 --use_amp
  ```
  若 16 能跑，再试 24、32，直到显存用到 7～7.5 GB 左右即可。

- **稍大一点的模型 + 中等 batch**  
  ```bash
  python scripts/train_with_real_data.py --dataset wikitext2 --d_model 384 --num_layers 4 --num_heads 6 --batch_size 8 --use_amp
  ```

建议：**先只改 `--batch_size`**（例如 8 → 16 → 24），用 nvidia-smi 看 **Memory-Usage** 和是否 OOM，再决定是否加大 d_model/num_layers。

---

## 四、一句话对应你的问题

- **GPU 在怎样工作？**  
  正在跑你的 Python 训练（PID 35372），**算力 87% 被占用、功耗拉满**，说明在认真算，不是「没用到 GPU」。

- **显存为什么看起来不够？**  
  因为当前 **batch_size=4、模型也不大**，所以只用了约 **2.2 GB / 8 GB**；想多占显存就**把 batch_size 调大**（例如 16 或 24），再观察 nvidia-smi 的 Memory-Usage。

- **现在 epoch 和 batch size 是多少？**  
  **num_epochs = 5**（脚本默认），**batch_size = 4**（你命令里写的）。
