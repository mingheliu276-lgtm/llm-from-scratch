# CUDA 基础语法教程 ✨

你好呀～这份教程会手把手带你搞懂：**在咱们这个项目里，GPU（CUDA）是怎么被用起来的**。不用怕，我们从「会用的语法」开始，一点点建立手感；你之前跑过的 `cuda_basics.py` 也会在这里被拆开讲清楚，方便你以后自己改、自己试。

---

## 一、先搞清两件事：CUDA 是什么？我们实际在写什么？

- **CUDA** 是 NVIDIA 给 GPU 做通用计算的一套东西：驱动、运行时、编程模型（线程、显存、kernel 等）。  
- **在我们项目里**，你写的都是 **Python + PyTorch**，并没有直接写 CUDA C++ 或 `.cu` 文件。PyTorch 在背后帮我们调用了 CUDA：你把张量放到 GPU 上、做运算，PyTorch 就会在 GPU 上执行。  
所以，**「CUDA 基础语法」在这里 = 在 PyTorch 里正确、安全地使用 GPU 的写法和概念**。本教程就围绕这个来。

---

## 二、核心语法 1：设备（Device）—— 数据在哪算？

在 PyTorch 里，**设备** 要么是 CPU，要么是某块 GPU（CUDA）。

```python
import torch

# 当前机器有没有可用 GPU？
torch.cuda.is_available()   # True / False

# 有几块 GPU？
torch.cuda.device_count()   # 例如 1

# 选一块设备来用（字符串写法，最常用）
device = "cuda" if torch.cuda.is_available() else "cpu"
# 或指定第 0 块 GPU
device = "cuda:0"
```

- **`"cpu"`**：在 CPU 上创建和计算。  
- **`"cuda"` 或 `"cuda:0"`**：在第 0 块 GPU 上。多卡时可以用 `"cuda:1"`、`"cuda:2"` 等。

**把张量放到指定设备上：**

```python
# 方法 1：创建时直接指定设备（推荐，少一次拷贝）
x = torch.randn(3, 4, device=device)

# 方法 2：先创建再搬过去
x = torch.randn(3, 4)
x = x.to(device)   # 或旧写法 x.cuda()，效果类似

# 从 GPU 搬回 CPU（例如要打印或存文件时）
x_cpu = x.cpu()
```

**小口诀：** 想在哪算，就 `.to(设备)` 或 `device=设备`；算完了要回 CPU 就 `.cpu()`。

---

## 三、核心语法 2：模型也要上 GPU

光把数据放 GPU 不够，**模型参数** 也得在同一块 GPU 上，否则会报错或偷偷在 CPU 上算。

```python
model = MyModel(...)
model = model.to(device)

# 之后输入也放到同一 device
inputs = inputs.to(device)
outputs = model(inputs)
```

训练脚本里常见的写法就是：`model.to(device)`，然后每个 batch 都 `batch.to(device)`。这样前向、反向都在 GPU 上完成。

---

## 四、核心语法 3：同步 —— 为什么有时要「等 GPU 算完」？

GPU 的运算是**异步**的：你发完一条「在 GPU 上算 A+B」的指令，CPU 可能立刻往下执行，而 GPU 还在后面算。所以如果你用 `time.time()` 在 CPU 上计时，不「等 GPU」的话，测到的时间会偏小甚至几乎为 0。

**让 CPU 等当前 GPU 上的任务都算完：**

```python
torch.cuda.synchronize()
```

所以像 `cuda_basics.py` 里那样，**计时前**先跑一遍「预热」再 `synchronize()`，**计时后**再 `synchronize()`，这样得到的才是 GPU 真正算完的时间。  
平时训练循环里一般不用手写 `synchronize()`，但**做性能测试、对比 CPU/GPU 速度时一定要用**，否则数字会失真。

---

## 五、核心语法 4：显存 —— 看了 nvidia-smi 之后怎么在代码里看？

除了用 `nvidia-smi` 看整张卡的显存，PyTorch 还能看「当前进程」用了多少：

```python
# 当前进程在默认 GPU(0) 上已经「分配」了多少显存（约等于真正在用）
torch.cuda.memory_allocated(0)   # 单位：字节

# PyTorch 为缓存分配的显存（可能还没全用上）
torch.cuda.memory_reserved(0)

# 转成 MB 方便看
print(f"已分配: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
```

**主动释放缓存（一般只在显存紧张或测显存时用）：**

```python
torch.cuda.empty_cache()
```

删掉不用的张量（`del big_tensor`）后，再调 `empty_cache()`，显存会更快还给系统。正常训练可以不用频繁调。

---

## 六、核心语法 5：常用 API 速查

| 你写的代码 | 作用 |
|------------|------|
| `torch.cuda.is_available()` | 当前环境是否能用 CUDA（有驱动 + 对应 PyTorch 版本）。 |
| `torch.cuda.device_count()` | 可见的 GPU 数量。 |
| `torch.cuda.get_device_name(0)` | 第 0 块 GPU 的名字（如 RTX 5060）。 |
| `tensor.to(device)` / `tensor.cuda()` | 把张量放到 GPU（或指定设备）。 |
| `tensor.cpu()` | 把张量从 GPU 搬回 CPU。 |
| `model.to(device)` | 把模型所有参数和 buffer 搬到指定设备。 |
| `torch.cuda.synchronize()` | CPU 等待当前 GPU 流上的操作完成（计时时必备）。 |
| `torch.cuda.memory_allocated(0)` | 当前进程在 GPU 0 上已分配显存（字节）。 |
| `torch.cuda.empty_cache()` | 释放未使用的显存缓存。 |

---

## 七、和 `cuda_basics.py` 的对应关系

你项目里的 `scripts/cuda_basics.py` 其实就是上面语法的「活用法」：

1. **环境检查**：`torch.cuda.is_available()`、`get_device_name`、`device_count`、显存与计算能力。  
2. **向量加法**：在 CPU 上创建 `a_cpu, b_cpu`，在 GPU 上创建 `a_gpu, b_gpu`（`.cuda()`），比较 `a+b` 的时间，并强调 **GPU 计时前后要 `synchronize()`**。  
3. **矩阵乘法**：大矩阵在 GPU 上 `torch.matmul`，同样用 `synchronize()` 计时，最后用 `empty_cache()` 做清理。  
4. **显存管理**：用 `memory_allocated` 看分配、`del` + `empty_cache()` 看释放。  
5. **梯度**：在 GPU 上创建 `requires_grad=True` 的张量，做运算后 `.backward()`，看 `.grad`；和训练时一样，只是小例子。

建议：**先按本教程把「设备、.to()、synchronize、显存」几条记牢**，再打开 `cuda_basics.py` 逐段看，你会觉得每一行都在复习上面的语法。

---

## 八、一条「不报错」的 GPU 流水线（模板）

把前面串成一段可复用的模板，方便你写自己的小实验或脚本：

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 数据上 GPU
x = torch.randn(100, 100, device=device)

# 若有模型，也要 to(device)
# model = MyModel().to(device)
# out = model(x)

# 若要计时，记得同步
# torch.cuda.synchronize()
# start = time.time()
# ... 计算 ...
# torch.cuda.synchronize()
# print(time.time() - start)
```

---

## 九、想再往深处走一点（可选）

- **多卡**：用 `device="cuda:0"` / `"cuda:1"` 把不同模型或数据放到不同卡；更进阶用 `torch.nn.DataParallel` 或分布式。  
- **混合精度**：咱们训练脚本里的 `--use_amp` 就是用 `torch.amp.autocast("cuda")` 等，在能用的地方用半精度，省显存、提速度。  
- **真正的 CUDA C++**：写 `.cu`、写 kernel、配置 block/thread 等，是另一门课；做 LLM 训练用 PyTorch 的这套「CUDA 语法」就够用了。

---

希望这份教程能让你觉得「GPU 没那么神秘，就是数据放上去、模型放上去、记得同步和显存」～  
接下来可以：再跑一遍 `python scripts/cuda_basics.py`，对照本文逐段看；或者在自己写的小脚本里故意用错 device（比如只把数据放 GPU、模型不放），看看报错长什么样，印象会更牢。  
如果你愿意，也可以说说你接下来想先练哪一块（比如：多卡、显存排查、计时），我们可以再单独写一小节实战练习。加油～ 💪
