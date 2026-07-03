import torch
import torch.nn as nn
from Embedders.token_embedder import TokenEmbedder

# ────────────────────────────────────────────────────────
# THE TRANSFORER BLOCK (Where the Pre-Norm lives)
# ────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)       # Pre-Norm 1
        self.attn = GQAEngine(n_embd) # GQA Engine
        self.ln_2 = nn.LayerNorm(n_embd)       # Pre-Norm 2
        self.ffn  = FeedForwardNetwork(n_embd) # Knowledge Bank

    def forward(self, x):
        # EXECUTION POINT: The data 'x' enters here from the Embedding layer.
        # It immediately runs through self.ln_1 before touching the Attention layer.
        x = x + self.attn(self.ln_1(x))  # <── First Pre-Norm executed here!
        
        # Second Pre-Norm execution before the FFN layer
        x = x + self.ffn(self.ln_2(x))
        return x

# ────────────────────────────────────────────────────────
# THE FULL LLM ENGINE (Assembling the pieces)
# ────────────────────────────────────────────────────────
class IceCreamSLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # This creates your stack of Transformer blocks (e.g., 6 layers)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        # 1. Convert raw input token integers to vectors
        x = self.token_embedding(idx) 
        
        # 2. Hand 'x' over to the Transformer layer pipeline
        for block in self.blocks:
            x = block(x) # When block loop index is 0, it jumps to the line marked above!
            
        # 3. Final normalization and output guessing
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits