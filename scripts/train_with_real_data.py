"""
使用真实数据集训练模型
支持WikiText-2等数据集
"""

import argparse
import torch
from pathlib import Path
import sys
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import SimpleGPT
from src.training.trainer import Trainer
from src.data.wikitext2 import load_wikitext2


def main():
    parser = argparse.ArgumentParser(description="使用真实数据集训练模型")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                       help="数据集 (默认 wikitext2)")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="检查点保存目录（AutoDL 建议 /root/autodl-tmp/checkpoints）")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("使用真实数据集训练模型")
    print("=" * 60)
    
    # 加载数据集（仅支持真实数据：WikiText-2）
    train_dataset, val_dataset, vocab_size = load_wikitext2(
        max_seq_len=args.max_seq_len
    )
    if train_dataset is None:
        return
    
    # 创建DataLoader（Linux/AutoDL 用 num_workers>0 加速，Windows 用 0）
    import platform
    num_workers = 4 if platform.system() == "Linux" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # 创建模型
    print(f"\n创建模型 (vocab_size={vocab_size})...")
    model = SimpleGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    )
    
    print(f"模型参数量: {model.count_parameters():,}")
    
    # 创建训练器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        use_amp=args.use_amp and device == "cuda",
        save_dir=Path(args.save_dir),
        config={
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "max_seq_len": args.max_seq_len,
        },
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train(num_epochs=args.num_epochs)
    
    print("\n训练完成!")
    print(f"模型保存在: {args.save_dir}/best_model.pt")
    print("\n下一步:")
    print("1. 使用 python scripts/inference.py 进行推理")
    print("2. 使用 python scripts/chat.py 进行对话")


if __name__ == "__main__":
    main()
