"""
WikiText-2 数据集加载
需要: pip install datasets transformers
"""

import torch
from pathlib import Path


def load_wikitext2(max_seq_len: int = 128):
    """
    加载 WikiText-2 数据集
    返回: (train_ds, val_ds, vocab_size) 或 (None, None, None) 若导入失败
    """
    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
        from torch.utils.data import Dataset
    except ImportError:
        print("请先安装: pip install datasets transformers")
        return None, None, None

    print("加载 WikiText-2 数据集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len + 1,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }

    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]

    class TokenizedDataset(Dataset):
        def __init__(self, dataset):
            self.data = dataset

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            input_ids = item["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            return input_ids

    train_ds = TokenizedDataset(train_dataset)
    val_ds = TokenizedDataset(val_dataset)

    print(f"训练样本数: {len(train_ds)}, 验证样本数: {len(val_ds)}")
    return train_ds, val_ds, len(tokenizer)
