"""
完整的Transformer模型实现
包含：FFN、Encoder 单层、完整 Encoder、以及 GPT 风格的语言模型（Decoder-only 简化版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 处理相对导入（支持直接运行和作为模块导入）
try:
    from .attention import MultiHeadAttention, PositionalEncoding
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.models.attention import MultiHeadAttention, PositionalEncoding


# =============================================================================
# 1. Feed-Forward Network（前馈子层）
# =============================================================================
# 论文里每个 Attention 层后面跟一个两层的 MLP：先升维到 d_ff，再压回 d_model。
# 公式：FFN(x) = Linear2( ReLU( Linear1(x) ) )，中间用 ReLU 做非线性。
# -----------------------------------------------------------------------------

class FeedForward(nn.Module):
    """前馈神经网络（FFN）：每个 token 独立做两次线性变换 + ReLU"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 先升维：d_model → d_ff（通常 d_ff = 4 * d_model）
        self.linear1 = nn.Linear(d_model, d_ff)
        # 再降回：d_ff → d_model，保持和主分支一致以便残差相加
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 形状不变：[B, seq_len, d_model] → [B, seq_len, d_model]
        # B: How many sentences.
        # seq_len: How many words in each sentence.
        # d_model: How many "features" or "attributes" each word has.
        # 注意：这里对“序列长度”和“batch”没有交互，每个位置独立计算
        return self.linear2(self.dropout(F.relu(self.linear1(x))))




# =============================================================================
# 2. Transformer Encoder 单层（一个 Block）
# =============================================================================
# 标准结构：Self-Attention → 残差 + LayerNorm → FFN → 残差 + LayerNorm。
# 残差连接让梯度更好传，LayerNorm 稳定训练。
# -----------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder 的一层：一个 Self-Attention 子层 + 一个 FFN 子层"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # LayerNorm：对最后一维（d_model）做归一化，均值 0 方差 1，可学习缩放和平移
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # ----- 子层1：Self-Attention + 残差 + Pre-LN -----
        # attn_output 与 x 同形状 [B, seq_len, d_model]
        attn_output = self.self_attn(x, mask)
        # 残差：x + dropout(attn_output)，再 LayerNorm（Pre-LN 写法：先加再 norm）
        x = self.norm1(x + self.dropout1(attn_output))

        # ----- 子层2：FFN + 残差 + Pre-LN -----
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


# =============================================================================
# 3. 完整 Transformer Encoder
# =============================================================================
# 输入：整型 token ids；输出：每个位置的 d_model 维向量。
# 流程：Embedding → 乘 scale → 位置编码 → 堆叠 N 个 Encoder 层。
# -----------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """完整 Encoder：把 [B, seq_len] 的 token id 变成 [B, seq_len, d_model] 的表示"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 词嵌入：把每个 token id 映射成 d_model 维向量；padding_idx 对应向量固定为 0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # 位置编码：给序列加上“第几个位置”的信息（sin/cos）
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # 堆叠 num_layers 个 Encoder 层；ModuleList 正确注册子模块
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] token ids
            mask: [batch_size, seq_len, seq_len] attention mask
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 输入嵌入：查表 [B, seq_len]→[B, seq_len, d_model]，乘 sqrt(d_model) 做 scale
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        if mask is None:
            mask = self._create_padding_mask(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x

    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """创建 padding mask（此处简化，返回 None 表示不遮住任何位置）"""
        return None


# =============================================================================
# 4. SimpleGPT：语言模型（Decoder-only 简化版）
# =============================================================================
# 用 Encoder 堆叠得到上下文表示，再用一个线性层把 d_model 映射到 vocab_size，
# 得到每个位置“下一个 token”的 logits，用于交叉熵损失。
# 注：完整 GPT 应在 Attention 里用因果 mask，此处用 Encoder 做简化。
# -----------------------------------------------------------------------------

class SimpleGPT(nn.Module):
    """GPT 风格语言模型：输入 token 序列，输出每个位置的下一 token logits"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 主体：Transformer Encoder（简化版 GPT 未用因果 mask，训练时可由外部传入 mask）
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        # 语言模型头：把每个位置的 d_model 维向量映射到 vocab_size 个 logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] token ids
            mask: 可选，[B, seq_len, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]，用于 CrossEntropyLoss
        """
        hidden_states = self.encoder(x, mask)
        logits = self.lm_head(hidden_states)
        return logits

    def count_parameters(self) -> int:
        """可训练参数总数。p.numel() 为张量元素个数，requires_grad 只计需要梯度的"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 整体数据流与形状速查（从 token id 到 logits）
# ---------------------------------------------------------------------------
# 输入 x:              [B, seq_len]  (token id)
# embedding(x):         [B, seq_len, d_model]
# * sqrt(d_model):      [B, seq_len, d_model]
# + pos_encoding:       [B, seq_len, d_model]
# 每层 EncoderLayer:    [B, seq_len, d_model] → [B, seq_len, d_model]
# lm_head(hidden):      [B, seq_len, vocab_size]  (logits)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("测试Transformer模型...")

    vocab_size = 10000
    batch_size = 2
    seq_len = 128

    model = SimpleGPT(
        vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
    )
    
    # 创建输入（随机token ids）
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    logits = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {model.count_parameters():,}")
    print("✓ Transformer测试通过!")
  
