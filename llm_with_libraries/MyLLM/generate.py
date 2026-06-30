import os
import argparse
import torch
from tokenizer import CustomTokenizer
from model import GPT

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='1M', choices=['1M', '10M', '30M', '100M'])
    parser.add_argument('--prompt', type=str, default="The king walked", help="Prompt text")
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    args, unknown = parser.parse_known_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(base_dir, "tokenizer.model")
    checkpoint_path = os.path.join(base_dir, "checkpoints", f"ckpt_{args.config}.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load model with correct device mapping
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['config']
    
    device = cfg.device
    print(f"Using device: {device}")
    
    # Initialize and load model weights
    model = GPT(cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Load trained BPE tokenizer
    tokenizer = CustomTokenizer(tokenizer_path)
    if tokenizer.tokenizer is None:
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}.")
        
    # Encode input prompt
    print(f"\nPrompt: '{args.prompt}'")
    input_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Autoregressive generation
    print("Generating...")
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        
    # Decode and print generated text
    output_ids = y[0].tolist()
    generated_text = tokenizer.decode(output_ids)
    print(f"\nGenerated Output:\n{generated_text}\n")

if __name__ == '__main__':
    generate()
