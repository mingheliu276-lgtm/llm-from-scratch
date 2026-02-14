"""
简单的对话脚本 - 与训练好的模型对话
注意：需要真实数据集训练的模型才能有意义的对话
"""

import argparse
import torch
from pathlib import Path
import sys
print(sys.path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference import load_model, generate_text


def main():
    parser = argparse.ArgumentParser(description="与模型对话")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 检查文件是否存在
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print("❌ 错误: 模型文件不存在")
        print("请先训练模型")
        return
    
    # 加载模型（checkpoint 若含 config 会自动用正确 vocab_size / max_seq_len）
    print("加载模型...")
    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    tokenizer = None
    if getattr(model, "vocab_size", 0) == 50257:
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("模型对话模式")
    print("=" * 60)
    print("输入问题，模型会生成回答")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break
            
            # 生成回答（WikiText-2 模型会用 GPT-2 tokenizer）
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_length=50,
                temperature=0.8,
                device=device,
                repetition_penalty=1.3,
            )
            
            print(f"模型: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n⚠️  重要提示:")
    print("当前模型使用虚拟数据集训练，无法回答真实问题")
    print("要获得有意义的对话，请:")
    print("1. 使用真实数据集训练: python scripts/train_with_real_data.py --dataset wikitext2")
    print("2. 或者使用更大的预训练模型（如GPT-2）进行微调")


if __name__ == "__main__":
    main()
