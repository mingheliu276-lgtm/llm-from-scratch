# 理解 Transformer 架构

你已经熟悉了 **Multi-Head Self-Attention**，本文在同一详细程度上把 **Transformer 整体架构**串起来：每个模块在算什么、用到了哪些 API、数据形状怎么变，以及和 Attention 的关系。

---

## 一、Transformer 在做什么（整体目标）

- **输入**：一串 **token id**，形状 `[batch_size, seq_len]`（例如一句话 tokenize 后的整数序列）。
- **输出**（本项目 SimpleGPT）：
  - 每个位置一个 **d_model 维向量**（中间表示），再通过一个线性层得到每个位置的 **vocab_size 个 logits**，用于预测「下一个 token」。
- **核心思想**：不用 RNN 的逐步递推，而是用 **Self-Attention** 让每个位置直接看到整段序列，再配合 **前馈网络（FFN）**、**残差** 和 **LayerNorm** 堆叠成「层」，多层堆叠成完整模型。

所以：**Attention 是 Transformer 里的一小块；Transformer = 嵌入 + 位置编码 + 很多个「Attention + FFN」层 + 最后的输出头。**

---

## 二、和 Attention 的关系（你已熟悉的部分）

在 `attention.py` 里你已经见过：

- **MultiHeadAttention**：输入 `[B, seq_len, d_model]`，输出同形状；内部做 Q/K/V 线性变换、多头、softmax 加权、对 V 求和、再线性输出。
- **PositionalEncoding**：给同一形状的向量加上 sin/cos 位置编码。

在 Transformer 里：

- **每个 Encoder 层**里会 **调用一次** `MultiHeadAttention`（即 Self-Attention），然后再过一个 **FeedForward**，并且都带 **残差 + LayerNorm**。
- **位置编码**在进入第一层之前加在嵌入上，所以之后每一层看到的输入都已经带位置信息。

可以简单记：**一层 = Self-Attention 子层 + FFN 子层**，每子层都是「子层输出 + 残差 → LayerNorm」。

---

## 三、各模块详解（对应代码里的谁）

### 1. 词嵌入（Embedding）

- **作用**：把离散的 token id 变成连续的 d_model 维向量，才能做后面的线性变换和 Attention。
- **代码**：`nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)`  
  - 相当于一张表：`embedding(id)` 取出第 `id` 行，形状从 `[B, seq_len]` → `[B, seq_len, d_model]`。
  - `padding_idx`：某个 id（例如 0）表示「空白/填充」，这一行被固定为 0 且不参与梯度，避免 padding 影响训练。
- **乘 `sqrt(d_model)`**：论文里为了和后面位置编码的尺度匹配，会对嵌入结果乘 \(\sqrt{d_{\text{model}}}\)。

### 2. 位置编码（PositionalEncoding）

- 你已经学过：用 sin/cos 给每个位置一个固定向量，和嵌入相加，这样模型能区分「第几个词」。
- 在 Transformer 里：**只加一次**，加在 `embedding(x) * sqrt(d_model)` 之后、进入第一层之前。

### 3. 前馈网络（FeedForward / FFN）

- **作用**：每个位置 **独立** 做两次线性变换 + 中间 ReLU，不跨位置、不跨 batch，只是把 d_model 维向量换一种表示。
- **公式**：  
  \(\text{FFN}(x) = W_2\, \text{ReLU}(W_1 x)\)  
  其中 \(W_1\) 把 d_model 映射到 d_ff（通常 d_ff = 4×d_model），\(W_2\) 再映射回 d_model。
- **代码**：`linear1(x)` → `ReLU` → `dropout` → `linear2`，输入输出形状都是 `[B, seq_len, d_model]`。

### 4. LayerNorm（层归一化）

- **作用**：对 **最后一维（d_model）** 做归一化：减均值、除标准差，再乘可学习缩放、加可学习平移，使每层的输入尺度更稳定，训练更容易。
- **代码**：`nn.LayerNorm(d_model)`，输入输出形状不变。

### 5. 残差连接（Residual Connection）

- **作用**：把「子层的输入」和「子层的输出」**相加** 再送入 LayerNorm，即 `x_new = LayerNorm(x + sublayer(x))`。这样梯度可以直接沿加法传回去，减轻深层网络的梯度消失。
- **代码**：例如 `self.norm1(x + self.dropout1(attn_output))`，就是 Attention 子层的残差 + Pre-LN。

### 6. Transformer Encoder 的一层（TransformerEncoderLayer）

- **顺序**：  
  1. Self-Attention：`attn_output = self.self_attn(x, mask)`  
  2. 残差 + LayerNorm：`x = norm1(x + dropout1(attn_output))`  
  3. FFN：`ffn_output = feed_forward(x)`  
  4. 残差 + LayerNorm：`x = norm2(x + dropout2(ffn_output))`  
  输入输出形状都是 `[B, seq_len, d_model]`。

### 7. 完整 Encoder（TransformerEncoder）

- **顺序**：  
  1. `embedding(x) * sqrt(d_model)`  
  2. 加位置编码  
  3. 若有 mask 则传给每一层（本项目简化时可为 None）  
  4. 重复 N 次：`x = layer(x, mask)`  
  输出形状：`[B, seq_len, d_model]`。

### 8. SimpleGPT（语言模型）

- **结构**：用一个 **TransformerEncoder** 得到 `hidden_states`，再用一个线性层 **lm_head** 把 d_model 映射到 vocab_size，得到 logits。
- **含义**：`logits[b, t, :]` 表示「在第 t 个位置，下一个 token 是词表里每个词」的未归一化分数，训练时和「下一个 token 的 id」做交叉熵。

---

## 四、用到的 API / 概念速查

| 名称 | 作用 |
|------|------|
| `nn.Embedding(num_embeddings, embedding_dim, padding_idx=?)` | 根据 id 查表得到向量；padding_idx 对应行固定为 0 且不更新。 |
| `nn.LayerNorm(normalized_shape)` | 对最后一维做归一化（减均值除标准差再仿射），形状不变。 |
| `nn.ModuleList([...])` | 把多个 `nn.Module` 放进列表并注册为子模块，方便 `for layer in self.layers` 和参数管理。 |
| `F.relu(x)` | 逐元素 \(\max(0, x)\)。 |
| `math.sqrt(d_model)` | 标量，嵌入后的缩放因子。 |
| `p.numel()` | 张量元素个数；`sum(p.numel() for p in model.parameters())` 即总参数量。 |

---

## 五、数据流与形状（和 Attention 的衔接）

下面从「整型 token」到「logits」走一遍，方便你在脑子里对上线。

1. **输入**  
   `x`: `[B, seq_len]`（token id）

2. **嵌入 + 缩放**  
   `embedding(x) * sqrt(d_model)` → `[B, seq_len, d_model]`

3. **位置编码**  
   `x + pe[:, :seq_len, :]`，形状仍为 `[B, seq_len, d_model]`

4. **每个 Encoder 层**（重复 N 次）  
   - 进入层时：`x` 为 `[B, seq_len, d_model]`  
   - Self-Attention：你学过的 MultiHeadAttention，输入输出都是 `[B, seq_len, d_model]`；内部 Q/K/V 的线性、多头、softmax、对 V 加权求和、输出线性，都在这个形状下进行。  
   - 残差 + LayerNorm：`x = norm1(x + dropout1(attn_output))`，形状不变。  
   - FFN：`[B, seq_len, d_model]` → `[B, seq_len, d_model]`。  
   - 残差 + LayerNorm：`x = norm2(x + dropout2(ffn_output))`，形状不变。  
   - 出层时：仍是 `[B, seq_len, d_model]`

5. **输出头**  
   `lm_head(hidden_states)`：`[B, seq_len, d_model]` → `[B, seq_len, vocab_size]`，这就是 **logits**，用于语言建模的 CrossEntropyLoss。

所以：**Attention 的输入/输出形状就是 Encoder 层里的「那一截」；整条链就是 嵌入 → 位置 → 多段「Attention + FFN」→ lm_head。**

---

## 六、本项目里的简化（和完整 GPT 的差别）

- **Encoder-only**：没有实现 Decoder 和 Cross-Attention，只堆叠 Encoder 层，适合「看整句预测下一个 token」的简化语言模型。
- **因果 mask**：完整 GPT 会在 Attention 里用**因果 mask**（当前位只能看过去），防止看到未来 token。本项目中 Encoder 默认不强制因果，若需要可由外部传入 mask。
- **Padding mask**：`_create_padding_mask` 当前返回 None，即没有把 padding 位置遮掉；若要严格按长度训练，可以在这里根据 token id 生成 mask 传给各层。

---

## 七、建议阅读顺序（配合代码）

1. 打开 `src/models/transformer.py`，从 **FeedForward** 看起（你已经会 Linear、ReLU、Dropout）。  
2. 再看 **TransformerEncoderLayer**：哪里调用了 Attention、哪里残差、哪里 LayerNorm。  
3. 再看 **TransformerEncoder**：嵌入、位置编码、`for layer in self.layers`。  
4. 最后看 **SimpleGPT**：encoder + lm_head，以及 `count_parameters`。

文件里已经加了和 `attention.py` 同风格的逐段注释，遇到不认识的 API 可以对照本文第四节。这样你可以把「Attention 机制」和「Transformer 架构」在脑子里连成一条线：Attention 是层内的一步，Transformer 是整条从 token id 到 logits 的管道。
