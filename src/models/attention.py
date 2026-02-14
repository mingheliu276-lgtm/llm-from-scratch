"""
Multi-Head Self-Attention 实现
这是Transformer的核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention机制
    
    Args:
        d_model: 模型维度（通常是512, 768, 1024等）
        num_heads: 注意力头数（必须是d_model的约数）
        dropout: Dropout概率
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        # 必须调用父类初始化，这样 PyTorch 才能正确追踪并训练本类里的 nn.Linear 等子模块
        super().__init__()
        # 多头要把 d_model 拆成 num_heads 份，每份 d_k 维，所以必须能整除
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        # 每个头负责的维度。例如 d_model=512, num_heads=8 → d_k=64
        self.d_k = d_model // num_heads

        # 四个线性层：输入 x 都是 [..., d_model]，输出也是 [..., d_model]
        # W_q: 把 x 变成 Query（“我在找什么”）
        self.W_q = nn.Linear(d_model, d_model)
        # W_k: 把 x 变成 Key（“我有什么可被查的”）
        self.W_k = nn.Linear(d_model, d_model)
        # W_v: 把 x 变成 Value（“查到我时取走的内容”）
        self.W_v = nn.Linear(d_model, d_model)
        # W_o: 多头拼起来之后再做一次线性变换，作为最终输出
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        # 缩放因子 = sqrt(d_k)。点积 Q·K 的方差随 d_k 增大，除以它让 softmax 前数值更稳定
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 None；mask[b,i,j]=0 表示位置 i 不能看位置 j
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 解包形状，第三维就是 d_model，用 _ 表示“这里不关心具体数字”
        batch_size, seq_len, _ = x.shape

        # ========== 1. 计算 Q, K, V ==========
        # 对同一批输入 x 做三次不同的线性变换，得到 Query、Key、Value
        # 形状都是 [batch_size, seq_len, d_model]，即每个时间步一个 d_model 维向量
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # ========== 2. 拆成“多头” ==========
        # 把最后一维 d_model 拆成 num_heads 份，每份 d_k 维，方便每个头独立做注意力
        # view: [B, seq_len, d_model] → [B, seq_len, num_heads, d_k]（元素顺序不变，只是重新解释形状）
        # transpose(1,2): 把“序列长度”和“头”两维互换 → [B, num_heads, seq_len, d_k]
        # 这样后面做 matmul 时，相当于对每个头单独做 [seq_len, d_k] @ [d_k, seq_len] 的注意力矩阵
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # ========== 3. 计算注意力分数（未归一化） ==========
        # Q: [B, num_heads, seq_len, d_k],  K: [B, num_heads, seq_len, d_k]
        # K.transpose(-2,-1): 只转置最后两维 → [B, num_heads, d_k, seq_len]
        # matmul(Q, K^T): 对每个头，seq_len 个 query 与 seq_len 个 key 两两点积
        # 结果 [B, num_heads, seq_len, seq_len]：scores[b,h,i,j] = 位置 i 对位置 j 的注意力分数
        # 除以 scale 是为了避免 d_k 很大时点积过大，导致 softmax 梯度接近 0
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # ========== 4. 应用 mask（可选） ==========
        # mask 中值为 0 的位置表示“不允许注意”，用 -inf 填充后，softmax 会把这些位置变成 0
        # 例如因果 mask：下三角为 1、上三角为 0，这样位置 i 只能看 0..i，不能看未来
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # ========== 5. 把分数变成“权重”（和为 1） ==========
        # dim=-1：在最后一维（对每个 query 对应的所有 key）上做 softmax，使每行和为 1
        # 得到 attn_weights[b,h,i,j] = 位置 i 分配给位置 j 的注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ========== 6. 用权重对 V 加权求和，得到每个位置的新表示 ==========
        # attn_weights: [B, num_heads, seq_len, seq_len],  V: [B, num_heads, seq_len, d_k]
        # matmul 时对最后一维（seq_len）做加权和：每个位置 i 得到 sum_j (weight[i,j] * V[j])
        # 结果 [B, num_heads, seq_len, d_k]
        attn_output = torch.matmul(attn_weights, V)

        # ========== 7. 把多个头拼回一个 d_model 维向量 ==========
        # transpose(1,2): [B, num_heads, seq_len, d_k] → [B, seq_len, num_heads, d_k]
        # contiguous(): transpose 后内存可能不连续，view 前需要先 contiguous
        # view: 把 num_heads*d_k 拼成 d_model → [B, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # ========== 8. 输出投影 ==========
        # 再做一次线性变换，形状保持 [B, seq_len, d_model]
        output = self.W_o(attn_output)

        return output


class PositionalEncoding(nn.Module):
    """
    位置编码 - 为序列添加位置信息
    Transformer 本身不区分顺序，用 sin/cos 给每个位置一个固定向量，让模型“知道”谁在前谁在后
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 先建一个空矩阵，后面按列填入 sin/cos。形状 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position: 位置下标 0, 1, 2, ..., max_len-1，形状 [max_len, 1]，方便和 div_term 广播
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 公式里 PE(pos, 2i) = sin(pos / 10000^(2i/d_model))，即不同维度用不同“频率”
        # 10000^(2i/d_model) = exp( (2i/d_model) * ln(10000) )，这里先算 1/10000^(2i/d_model)
        # torch.arange(0, d_model, 2) 得到 [0,2,4,...,d_model-2]，对应维度下标 2i
        # (-math.log(10000.0) / d_model) 是 -ln(10000)/d_model，乘上 2i 后 exp 得到 1/10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数列（0,2,4,...）用 sin(position * div_term)；奇数列（1,3,5,...）用 cos
        # position [max_len,1] * div_term [d_model/2] 广播成 [max_len, d_model/2]，再写入 pe 的偶数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加一维 batch： [max_len, d_model] → [1, max_len, d_model]，这样和 x 相加时自动沿 batch 广播
        pe = pe.unsqueeze(0)
        # 不参与梯度、但会随模型搬到 GPU；保存为 buffer 而不是 parameter，因为位置编码是固定的
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        返回同形状，每个位置加上了对应的位置编码向量
        """
        # self.pe 是 [1, max_len, d_model]，取前 seq_len 个位置与当前序列对齐
        # x.size(1) 就是 seq_len；加完后 dropout 一下
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 形状变化速查（forward 里张量形状的完整流程）
# ---------------------------------------------------------------------------
# 输入 x:              [B, seq_len, d_model]
# Q/K/V:               [B, seq_len, d_model]
# view + transpose 后:  [B, num_heads, seq_len, d_k]
# scores:              [B, num_heads, seq_len, seq_len]   (Q @ K^T)
# attn_weights:        [B, num_heads, seq_len, seq_len]   (softmax)
# attn_output:         [B, num_heads, seq_len, d_k]       (weights @ V)
# transpose+view 后:   [B, seq_len, d_model]
# 最终 output:         [B, seq_len, d_model]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("测试Multi-Head Attention...")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    output = attn(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ Attention测试通过!")

    print("\n测试Positional Encoding...")
    pos_enc = PositionalEncoding(d_model=d_model)
    x_with_pos = pos_enc(x)
    print(f"添加位置编码后形状: {x_with_pos.shape}")
    print("✓ Positional Encoding测试通过!")
