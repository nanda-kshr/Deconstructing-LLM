import os
import torch
from torch.utils.data import Dataset
import numpy as np

class TokenizedDataset(Dataset):
    """
    Efficient Dataset class loading BPE-encoded token IDs from a numpy binary file.
    """
    def __init__(self, data_file, block_size):
        # Using numpy memory-mapping or direct load to save memory
        self.data = np.load(data_file, mmap_mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Slice input and shifted target
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64))
        return x, y

def prepare_data(corpus_path, tokenizer, output_dir, val_split=0.1):
    """
    Tokenizes raw text corpus and splits it into train/val numpy binaries.
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Tokenizing corpus of size {len(text)/1e6:.2f}M characters...")
    token_ids = tokenizer.encode(text)
    
    # vocab_size <= 65535 is guaranteed to fit in uint16, saving significant space
    token_ids_arr = np.array(token_ids, dtype=np.uint16)
    
    n = len(token_ids_arr)
    val_size = int(n * val_split)
    train_ids = token_ids_arr[:-val_size]
    val_ids = token_ids_arr[-val_size:]
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.npy')
    val_path = os.path.join(output_dir, 'val.npy')
    
    np.save(train_path, train_ids)
    np.save(val_path, val_ids)
    
    print(f"Saved tokenized data to {output_dir}")
    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")
    return train_path, val_path
