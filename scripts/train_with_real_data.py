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


def load_wikitext2(vocab_size: int = 10000, max_seq_len: int = 128):
    """
    加载WikiText-2数据集
    需要先安装: pip install datasets
    """
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
    except ImportError:
        print("请先安装: pip install datasets transformers")
        return None, None
    
    print("加载WikiText-2数据集...")
    
    # 加载数据集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # 使用GPT-2的tokenizer（也可以自己实现简单的tokenizer）
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # 对文本进行tokenization
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len + 1,  # +1 for labels
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0)
        }
    
    # 过滤空文本
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    
    # 转换为PyTorch Dataset
    from torch.utils.data import Dataset
    
    class TokenizedDataset(Dataset):
        def __init__(self, hf_dataset):
            self.data = hf_dataset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            # 必须返回单个 tensor，否则 DataLoader 默认 collate 会把
            # 「4 个 list(129)」变成 129 个 tensor(4)，导致 batch 变成 129 元组
            input_ids = item["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            return input_ids
    
    train_ds = TokenizedDataset(train_dataset)
    val_ds = TokenizedDataset(val_dataset)
    
    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)}")
    
    return train_ds, val_ds, len(tokenizer)


def main():
    parser = argparse.ArgumentParser(description="使用真实数据集训练模型")
    parser.add_argument("--dataset", type=str, default="wikitext2", 
                       choices=["wikitext2", "dummy"],
                       help="数据集选择")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=128)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("使用真实数据集训练模型")
    print("=" * 60)
    
    # 加载数据集
    if args.dataset == "wikitext2":
        train_dataset, val_dataset, vocab_size = load_wikitext2(
            max_seq_len=args.max_seq_len
        )
        if train_dataset is None:
            return
    else:
        # 使用虚拟数据集
        from scripts.train import create_dummy_dataset
        train_dataset = create_dummy_dataset(
            vocab_size=10000,
            num_samples=1000,
            seq_len=args.max_seq_len
        )
        val_dataset = create_dummy_dataset(
            vocab_size=10000,
            num_samples=100,
            seq_len=args.max_seq_len
        )
        vocab_size = 10000
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
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
        save_dir=Path("checkpoints"),
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
    print(f"模型保存在: checkpoints/best_model.pt")
    print("\n下一步:")
    print("1. 使用 python scripts/inference.py 进行推理")
    print("2. 使用 python scripts/chat.py 进行对话")


if __name__ == "__main__":
    main()
