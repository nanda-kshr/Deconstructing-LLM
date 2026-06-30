import os
import time
import math
import argparse
import torch
from torch.utils.data import DataLoader
from config import configs
from tokenizer import CustomTokenizer
from dataset import TokenizedDataset, prepare_data
from model import GPT

def get_lr(step, max_steps, warmup_steps, lr):
    """
    Calculates learning rate with a linear warmup followed by a cosine decay.
    """
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step > max_steps:
        return lr * 0.1
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr * (0.1 + 0.9 * coeff)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='1M', choices=['1M', '10M', '30M', '100M'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    args, unknown = parser.parse_known_args() # Use parse_known_args for robustness in notebooks or run modes

    cfg = configs[args.config]
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        
    print(f"Using configuration: {args.config} on device {cfg.device}")
    
    # Paths (relative to MyLLM directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(base_dir, "corpus.txt")
    tokenizer_path = os.path.join(base_dir, "tokenizer.model")
    data_dir = os.path.join(base_dir, "data")
    
    # 1. Download/create corpus if not exists
    if not os.path.exists(corpus_path):
        print(f"Corpus {corpus_path} not found. Creating a tiny dummy corpus...")
        try:
            import requests
            print("Downloading Tiny Shakespeare corpus...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            r = requests.get(url)
            with open(corpus_path, "w", encoding='utf-8') as f:
                f.write(r.text)
            print("Download complete.")
        except Exception as e:
            print(f"Could not download, writing local mock corpus: {e}")
            mock_text = "The cat sat on the mat.\n" * 10000
            with open(corpus_path, "w", encoding='utf-8') as f:
                f.write(mock_text)
                
    # 2. Train tokenizer if not exists
    tokenizer = CustomTokenizer(tokenizer_path)
    if tokenizer.tokenizer is None:
        print("Training custom BPE tokenizer...")
        tokenizer.train(corpus_path, cfg.vocab_size, tokenizer_path)
        tokenizer = CustomTokenizer(tokenizer_path)
        
    # Ensure config vocab size matches the actual tokenizer vocab size
    cfg.vocab_size = tokenizer.tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {cfg.vocab_size}")

    # 3. Prepare data binaries
    train_npy = os.path.join(data_dir, 'train.npy')
    val_npy = os.path.join(data_dir, 'val.npy')
    if not os.path.exists(train_npy) or not os.path.exists(val_npy):
        print("Preparing tokenized data binaries...")
        prepare_data(corpus_path, tokenizer, data_dir)
        
    # 4. DataLoaders
    train_dataset = TokenizedDataset(train_npy, cfg.block_size)
    val_dataset = TokenizedDataset(val_npy, cfg.block_size)
    
    pin_mem = (cfg.device in ['cuda', 'mps'])
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin_mem, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_mem, num_workers=0)

    # 5. Initialize Model
    model = GPT(cfg)
    model.to(cfg.device)
    
    # 6. Configure Optimizer
    device_type = 'cuda' if 'cuda' in cfg.device else ('mps' if 'mps' in cfg.device else 'cpu')
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
    
    start_epoch = 0
    best_val_loss = float('inf')
    step = 0
    
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{args.config}.pt")
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed at epoch {start_epoch}, step {step}")

    # Estimate training length
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1) # 10% warmup
    
    print(f"Total training epochs: {args.epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total target steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            step += 1
            x, y = x.to(cfg.device), y.to(cfg.device)
            
            # LR Schedule step
            lr = get_lr(step, total_steps, warmup_steps, cfg.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            # Forward and backward
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            
            # Print loss
            if step % 20 == 0 or batch_idx == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Step {step}/{total_steps} | Batch {batch_idx}/{steps_per_epoch} | Loss {loss.item():.4f} | LR {lr:.2e}")
                
            # Evaluate validation loss
            if step % 200 == 0 or step == total_steps:
                model.eval()
                val_loss = 0.0
                val_steps = min(50, len(val_loader))
                
                with torch.no_grad():
                    for val_idx, (vx, vy) in enumerate(val_loader):
                        if val_idx >= val_steps:
                            break
                        vx, vy = vx.to(cfg.device), vy.to(cfg.device)
                        _, vloss = model(vx, vy)
                        val_loss += vloss.item()
                val_loss /= val_steps
                print(f"--> Validation Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'best_val_loss': best_val_loss,
                        'config': cfg
                    }
                    torch.save(checkpoint, checkpoint_path)
                    print(f"--> Saved best checkpoint to {checkpoint_path}")
                model.train()
                
        print(f"Epoch {epoch+1} finished in {time.time() - epoch_start_time:.2f}s")

if __name__ == '__main__':
    train()
