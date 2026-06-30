import torch
from dataclasses import dataclass

@dataclass
class LLMConfig:
    # Model architecture
    vocab_size: int = 16000
    block_size: int = 512      # Context length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = False         # True: bias in Linears and LayerNorms, like GPT-2. False: modern standard (faster)
    
    # Training parameters
    batch_size: int = 32       # Batch size per step
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0     # Clip gradients at this value, or 0.0 for no clipping
    
    # Device configuration
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

# Preset configs for progression
configs = {
    "1M": LLMConfig(
        vocab_size=8000,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        batch_size=64,
        learning_rate=6e-4
    ),
    "10M": LLMConfig(
        vocab_size=16000,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        batch_size=32,
        learning_rate=3e-4
    ),
    "30M": LLMConfig(
        vocab_size=16000,
        block_size=512,
        n_layer=12,
        n_head=8,
        n_embd=512,
        batch_size=16,
        learning_rate=3e-4
    ),
    "100M": LLMConfig(
        vocab_size=16000,
        block_size=512,
        n_layer=12,
        n_head=12,
        n_embd=768,
        batch_size=8,          # Kept small for MacBook M4 Air memory limit during training
        learning_rate=3e-4
    )
}
