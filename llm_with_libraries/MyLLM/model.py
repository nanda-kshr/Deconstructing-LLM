import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer import Block, LayerNorm

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (saves params and links representation spaces)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Total model parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        
        # Forward token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through transformer layers
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x) # (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: slice logits only for the last token to be faster
            logits = self.lm_head(x[:, [-1], :]) # (b, 1, vocab_size)
            loss = None
            
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Splits model parameters into two groups: decayed (weight matrices) and non-decayed (biases, layernorms).
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Decaying 2D weight tensors, not 1D (biases, layer norm parameters)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Decayed parameters: {len(decay_params)} tensors, {num_decay_params:,} params")
        print(f"Non-decayed parameters: {len(nodecay_params)} tensors, {num_nodecay_params:,} params")
        
        # Fused AdamW if available
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device_type == 'cuda' or device_type == 'mps')
        
        # Note: torch.optim.AdamW might not support fused on mps depending on version, so fallback to False on mps if error occurs
        extra_args = dict(fused=True) if use_fused and device_type == 'cuda' else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text starting from seed token sequence idx of shape (b, t).
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
