"""
训练器 - 实现完整的训练循环
包含前向传播、反向传播、优化器更新、检查点保存等
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import time
from pathlib import Path

# =============================================================================
# 混合精度训练辅助函数
# =============================================================================
# PyTorch 2.0+ 推荐用 torch.amp.autocast，旧版用 torch.cuda.amp.autocast
# 这个函数自动选择可用的 API，避免 FutureWarning
# -----------------------------------------------------------------------------

def _autocast(use_amp: bool):
    """返回混合精度训练的上下文管理器（autocast），优先用新 API"""
    if getattr(torch, "amp", None) is not None:
        return torch.amp.autocast("cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


# =============================================================================
# Trainer 类：完整的训练循环管理器
# =============================================================================
# 负责：数据加载 → 前向 → 损失 → 反向 → 优化器更新 → 验证 → 保存检查点
# 支持混合精度（FP16）、梯度累积、梯度裁剪、自动保存最佳模型
# -----------------------------------------------------------------------------

class Trainer:
    """
    大模型训练器
    
    支持：
    - 混合精度训练（FP16）：用半精度加速、省显存
    - 梯度累积：小 batch 也能模拟大 batch 的效果
    - 梯度裁剪：防止梯度爆炸
    - 模型检查点保存：训练中断后可恢复
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        use_amp: bool = True,  # 混合精度训练
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,  # 梯度裁剪
        save_dir: Path = Path("checkpoints"),
        config: Optional[dict] = None,  # 训练时的模型配置，会写入 checkpoint 供推理加载
    ):
        # ----- 模型与数据 -----
        # model.to(device)：把模型所有参数和 buffer 搬到指定设备（CPU/GPU）
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = Path(save_dir)
        # mkdir(parents=True, exist_ok=True)：创建目录，parents 表示创建父目录，exist_ok 表示已存在不报错
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config = config if config is not None else {}

        # ----- 优化器 -----
        # AdamW：Adam 的改进版，weight_decay 是 L2 正则化系数（防止过拟合）
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),  # 要优化的参数列表
                lr=1e-4,  # 学习率
                weight_decay=0.01,  # 权重衰减（L2 正则）
            )
        else:
            self.optimizer = optimizer

        # ----- 损失函数 -----
        # CrossEntropyLoss：多分类交叉熵，ignore_index=0 表示 label 为 0 的位置不参与损失计算（通常是 padding）
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            self.criterion = criterion

        # ----- 混合精度训练的 scaler -----
        # GradScaler：在 FP16 训练时动态缩放梯度，避免下溢（梯度太小变成 0）
        if use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except AttributeError:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # ----- 训练状态 -----
        self.global_step = 0  # 全局步数（所有 epoch 累计）
        self.epoch = 0  # 当前 epoch
        self.best_val_loss = float("inf")  # 最佳验证损失（用于保存最佳模型）
        
    def train_epoch(self) -> float:
        """
        训练一个 epoch：遍历所有训练 batch，做前向、反向、更新参数
        
        Returns:
            该 epoch 的平均训练损失
        """
        # model.train()：切换到训练模式（启用 dropout、batch norm 的训练行为）
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # ========== 1. 准备数据（兼容多种 batch 格式） ==========
            # DataLoader 返回的 batch 可能是：
            # - (tensor,)：TensorDataset 单个 tensor，需自己生成 labels（语言建模）
            # - (input_ids, labels)：两个 tensor 的 tuple，直接可用
            # - tensor：直接是 tensor（较少见）
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    # 两个 tensor：input_ids 和 labels
                    input_ids, labels = batch
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                elif len(batch) == 1:
                    # 单个 tensor：语言建模任务，labels 是 input_ids 向右移一位
                    original_input_ids = batch[0].to(self.device)
                    # [B, seq_len] → [B, seq_len-1]：去掉最后一个 token 作为输入
                    input_ids = original_input_ids[:, :-1]
                    # [B, seq_len] → [B, seq_len-1]：去掉第一个 token 作为标签（预测下一个）
                    labels = original_input_ids[:, 1:].contiguous()
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            else:
                # 直接是 tensor：同样按语言建模处理（与 len(batch)==1 一致）
                original_input_ids = batch.to(self.device)
                input_ids = original_input_ids[:, :-1]
                labels = original_input_ids[:, 1:].contiguous()

            # ========== 2. 前向传播 + 计算损失（混合精度） ==========
            # autocast：在 FP16 训练时，自动把部分操作转为半精度，加速且省显存
            with _autocast(self.use_amp):
                # 前向：输入 [B, seq_len-1]，输出 [B, seq_len-1, vocab_size]
                logits = self.model(input_ids)

                # 计算损失：CrossEntropyLoss 需要 [N, vocab_size] 和 [N] 两个张量
                # view(-1, vocab_size)：把 [B, seq_len-1, vocab_size] 展成 [B*(seq_len-1), vocab_size]
                # view(-1)：把 [B, seq_len-1] 展成 [B*(seq_len-1)]
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )

                # 梯度累积：loss 除以累积步数，这样多次 backward 的梯度累加后等于一次大 batch
                loss = loss / self.gradient_accumulation_steps

            # ========== 3. 反向传播 ==========
            # 混合精度时用 scaler.scale(loss).backward()：先缩放 loss，再反向传播
            # 普通训练直接用 loss.backward()：计算所有参数的梯度
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # ========== 4. 更新参数（每 accumulation_steps 步更新一次） ==========
            # 梯度累积：每 N 步才真正更新一次参数，模拟 batch_size * N 的效果
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # scaler.unscale_：在梯度裁剪前，先把 scaler 缩放的梯度还原
                    self.scaler.unscale_(self.optimizer)
                    # clip_grad_norm_：梯度裁剪，防止梯度爆炸（把所有参数的梯度范数限制在 max_grad_norm）
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    # scaler.step：更新参数（内部会再次缩放）
                    self.scaler.step(self.optimizer)
                    # scaler.update：更新 scaler 的缩放因子（根据是否有 inf/NaN）
                    self.scaler.update()
                else:
                    # 普通训练：直接裁剪梯度、更新参数
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                # zero_grad：清空梯度，为下一个 batch 准备
                self.optimizer.zero_grad()
                self.global_step += 1

            # ========== 5. 记录损失（用于计算 epoch 平均） ==========
            # loss.item()：把标量 tensor 转成 Python float
            # 乘以 accumulation_steps：因为之前 loss 被除了，这里乘回来得到「真实损失」
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # 打印进度（每 10 个 batch 打印一次）
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {self.epoch}, Batch {batch_idx+1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        验证：在验证集上评估模型，不更新参数
        
        @torch.no_grad()：装饰器，禁用梯度计算（节省显存、加速）
        
        Returns:
            验证集上的平均损失
        """
        if self.val_loader is None:
            return 0.0

        # model.eval()：切换到评估模式（禁用 dropout、batch norm 用统计值而非当前 batch）
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # 数据准备逻辑与 train_epoch 相同
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    input_ids, labels = batch
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                elif len(batch) == 1:
                    input_ids = batch[0].to(self.device)
                    labels = input_ids[:, 1:].contiguous()
                    input_ids = input_ids[:, :-1]
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            else:
                input_ids = batch.to(self.device)
                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1]

            # 前向传播（混合精度）+ 计算损失（不反向传播）
            with _autocast(self.use_amp):
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """
        保存检查点：模型参数、优化器状态、训练进度等
        
        Args:
            filename: 文件名（默认按 epoch 和 step 生成）
            is_best: 是否为最佳模型（验证损失最低），会额外保存到 best_model.pt
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}_step_{self.global_step}.pt"

        # state_dict()：返回模型/优化器/scaler 的参数字典（不含模型结构，只有权重）
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),  # 模型所有参数的权重
            "optimizer_state_dict": self.optimizer.state_dict(),  # 优化器状态（动量等）
            "best_val_loss": self.best_val_loss,
            "config": self.config,  # 模型配置（vocab_size、max_seq_len 等），推理时用
        }

        # 混合精度训练的 scaler 状态也要保存（缩放因子等）
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        filepath = self.save_dir / filename
        # torch.save：保存 checkpoint 字典到 .pt 文件（PyTorch 的 pickle 格式）
        torch.save(checkpoint, filepath)
        print(f"检查点已保存: {filepath}")

        # 如果是当前最佳模型，额外保存一份到 best_model.pt（推理脚本默认加载这个）
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")
    
    def load_checkpoint(self, filepath: Path):
        """
        加载检查点：恢复模型参数、优化器状态、训练进度
        
        Args:
            filepath: 检查点文件路径
        """
        # torch.load：从 .pt 文件加载字典
        # map_location=self.device：把加载的 tensor 放到指定设备（例如从 CPU checkpoint 加载到 GPU）
        checkpoint = torch.load(filepath, map_location=self.device)
        # load_state_dict：用 checkpoint 里的权重覆盖当前模型的参数
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        # get('best_val_loss', float('inf'))：如果 checkpoint 里没有这个 key，用默认值
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        # 恢复 scaler 状态（如果有）
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"检查点已加载: {filepath}")
    
    def train(self, num_epochs: int, save_every: int = 1):
        """
        完整训练流程：循环 num_epochs 次，每次训练 + 验证 + 保存
        
        Args:
            num_epochs: 训练轮数
            save_every: 每 N 个 epoch 保存一次检查点（默认每轮都保存）
        """
        print(f"开始训练，设备: {self.device}")
        # sum(p.numel() for p in ...)：统计所有可训练参数的总数
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()

            # ========== 训练一个 epoch ==========
            train_loss = self.train_epoch()

            # ========== 验证 ==========
            val_loss = self.validate()

            epoch_time = time.time() - start_time

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"耗时: {epoch_time:.2f}秒\n")

            # ========== 保存检查点 ==========
            # 判断是否为最佳模型（验证损失最低）
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # 按 save_every 的频率保存（默认每轮都保存）
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(is_best=is_best)


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.transformer import SimpleGPT
    
    print("测试Trainer...")
    
    # 创建模型
    model = SimpleGPT(vocab_size=10000, d_model=256, num_heads=4, num_layers=2)
    
    # 创建虚拟数据
    from torch.utils.data import TensorDataset, DataLoader
    dummy_data = torch.randint(1, 10000, (100, 128))  # 100个样本，每个128 tokens
    dataset = TensorDataset(dummy_data)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=torch.cuda.is_available(),
    )
    
    print("✓ Trainer初始化成功!")
    print(f"使用设备: {trainer.device}")
    print(f"混合精度训练: {trainer.use_amp}")
