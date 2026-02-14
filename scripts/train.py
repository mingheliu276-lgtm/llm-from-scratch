"""
训练脚本 - 从零开始训练大模型
用法: python scripts/train.py --config configs/small_model.yaml
"""

import argparse
import torch
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import SimpleGPT
from src.training.trainer import Trainer
from torch.utils.data import DataLoader


def create_dummy_dataset(vocab_size: int = 10000, num_samples: int = 1000, seq_len: int = 128):
    """创建虚拟数据集用于测试"""
    from torch.utils.data import TensorDataset
    
    # 生成随机token序列
    data = torch.randint(1, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(data)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="训练Transformer模型")
    parser.add_argument("--vocab_size", type=int, default=10000, help="词汇表大小")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer层数")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN维度")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda/cpu/auto)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="检查点保存目录")
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("大模型训练 - 从零开始")
    print("=" * 60)
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)
    
    # 创建模型
    print("\n创建模型...")
    model = SimpleGPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    )
    
    num_params = model.count_parameters()
    print(f"模型参数量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 估算显存使用（粗略）
    if device == "cuda":
        # 每个参数约4字节（FP32），加上激活值等
        estimated_memory_mb = (num_params * 4 * 2) / 1e6  # 参数 + 梯度
        print(f"估算显存使用: ~{estimated_memory_mb:.0f} MB (仅参数)")
    
    # 创建数据集
    print("\n创建数据集...")
    print("  [说明] 使用随机虚拟数据，仅用于验证训练流程，不能学到真实语言。")
    print("         损失含义及为何约 9.2 属正常，见项目根目录「理解训练输出.md」。")
    train_dataset = create_dummy_dataset(
        vocab_size=args.vocab_size,
        num_samples=1000,
        seq_len=args.max_seq_len,
    )
    val_dataset = create_dummy_dataset(
        vocab_size=args.vocab_size,
        num_samples=100,
        seq_len=args.max_seq_len,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows上设为0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # 创建训练器（保存 config 便于推理时自动匹配模型结构）
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        use_amp=args.use_amp and device == "cuda",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_dir=Path(args.save_dir),
        config={
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "max_seq_len": args.max_seq_len,
        },
    )
    
    # 开始训练
    print("\n开始训练...")
    print("-" * 60)
    trainer.train(num_epochs=args.num_epochs)
    
    print("\n训练完成!")
    print(f"最佳验证损失: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
