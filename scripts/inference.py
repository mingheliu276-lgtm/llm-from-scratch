"""
模型推理脚本 - 使用训练好的模型生成文本
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import SimpleGPT


def load_model(
    checkpoint_path: Path,
    device: str = "cuda",
    vocab_size: int = 10000,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    max_seq_len: int = 512,
):
    """
    加载训练好的模型。
    若 checkpoint 里带有 config，会优先使用；否则从 state_dict 里推断 vocab_size / max_seq_len，兼容旧检查点。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict") or checkpoint

    # 优先使用 checkpoint 里保存的 config
    cfg = checkpoint.get("config") or {}
    if cfg:
        vocab_size = cfg.get("vocab_size", vocab_size)
        d_model = cfg.get("d_model", d_model)
        num_heads = cfg.get("num_heads", num_heads)
        num_layers = cfg.get("num_layers", num_layers)
        max_seq_len = cfg.get("max_seq_len", max_seq_len)
    else:
        # 旧检查点没有 config：从 state_dict 推断，避免 shape 不匹配
        if "encoder.embedding.weight" in state:
            vocab_size = state["encoder.embedding.weight"].shape[0]
        if "encoder.pos_encoding.pe" in state:
            max_seq_len = state["encoder.pos_encoding.pe"].shape[1]
        if "encoder.embedding.weight" in state:
            d_model = state["encoder.embedding.weight"].shape[1]

    model = SimpleGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✓ 模型已加载: {checkpoint_path}")
    print(f"  vocab_size={vocab_size}, max_seq_len={max_seq_len}")
    print(f"  训练轮数: {checkpoint.get('epoch', '?')}")
    print(f"  最佳验证损失: {checkpoint.get('best_val_loss', '?'):.4f}")

    return model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    device: str = "cuda",
    repetition_penalty: float = 1.2,
):
    """
    生成文本。
    repetition_penalty > 1 会降低「刚出现过的 token」再被采样的概率，减轻 "are are are" 式循环。
    """
    model.eval()

    if tokenizer is None:
        vocab = {chr(i): i for i in range(1000, 10000)}
        prompt_ids = [vocab.get(c, 1) for c in prompt[:20]]
        prompt_ids = torch.tensor([prompt_ids], device=device)
    else:
        encoded = tokenizer.encode(prompt)
        prompt_ids = torch.tensor([encoded], device=device)

    generated = prompt_ids.clone()
    eos_id = getattr(tokenizer, "eos_token_id", 0) if tokenizer is not None else 0

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)
            next_token_logits = logits[0, -1, :].float().clone()

            # 重复惩罚：对已出现过的 token 降低 logits，减轻 "are are are" / "I I I" 循环
            if repetition_penalty != 1.0 and generated.shape[1] > 0:
                seen = generated[0].tolist()
                for token_id in set(seen):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == eos_id or next_token.item() == 0:
                break

    if tokenizer is None:
        result = "".join([chr(max(32, min(126, id.item()))) for id in generated[0]])
    else:
        result = tokenizer.decode(generated[0].cpu().tolist())

    return result


def main():
    parser = argparse.ArgumentParser(description="模型推理")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                       help="模型检查点路径")
    parser.add_argument("--prompt", type=str, default="Hello",
                       help="输入提示")
    parser.add_argument("--max_length", type=int, default=50,
                       help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="温度参数（越高越随机）")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                       help="重复惩罚，>1 可减轻「同一个词一直重复」")
    parser.add_argument("--vocab_size", type=int, default=10000,
                       help="词汇表大小（需与训练时一致）")
    parser.add_argument("--d_model", type=int, default=256,
                       help="模型维度")
    parser.add_argument("--num_heads", type=int, default=4,
                       help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Transformer层数")
    parser.add_argument("--max_seq_len", type=int, default=512,
                       help="最大序列长度（若 checkpoint 含 config 会优先用 config）")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 错误: 检查点文件不存在: {checkpoint_path}")
        print("\n请先训练模型:")
        print("  python scripts/train.py --d_model 256 --num_layers 2 --use_amp")
        return
    
    print("加载模型...")
    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
    )

    # 用 GPT-2 词表（50257）训练的模型需用同一 tokenizer 编解码
    tokenizer = None
    if getattr(model, "vocab_size", 0) == 50257:
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            print("  使用 GPT-2 tokenizer 进行编解码")
        except Exception:
            print("  未安装 transformers，将使用简单编码，生成质量可能较差")

    print(f"\n输入提示: {args.prompt}")
    print("生成中...")

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        device=device,
        repetition_penalty=args.repetition_penalty,
    )
    
    print(f"\n生成结果:")
    print("-" * 60)
    print(generated)
    print("-" * 60)
    
    print("\n⚠️  注意:")
    print("如果使用虚拟数据集训练的模型，生成的内容是随机的")
    print("要生成有意义的文本，请使用真实数据集训练:")
    print("  python scripts/train_with_real_data.py --dataset wikitext2")


if __name__ == "__main__":
    main()
