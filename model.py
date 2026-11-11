import math

import torch
from torch import nn
from torch.nn import functional as F

from attention import MultiHeadAttention
from configs import GPTConfig, get_config


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        assert ndim > 0, f"ndim must be positive, got {ndim}"
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        assert x.size(-1) == self.weight.numel(), (
            f"LayerNorm expected last dim {self.weight.numel()}, got {x.size(-1)}"
        )
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_intermediate > 0
        self.config = config
        hidden = config.n_intermediate * config.n_embed
        self.c_fc = nn.Linear(config.n_embed, hidden)
        self.c_proj = nn.Linear(hidden, config.n_embed)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        assert x.size(-1) == self.config.n_embed
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embed)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        residual = x
        x = x + self.attn(self.ln_1(x))
        assert x.shape == residual.shape
        x = x + self.mlp(self.ln_2(x))
        assert x.shape == residual.shape
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_layers > 0
        assert config.vocab_size > 0
        assert config.max_length > 0
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        self.wpe = nn.Embedding(config.max_length, config.n_embed)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = LayerNorm(config.n_embed)

    def forward(self, idx):
        assert idx.dim() == 2, f"expected (B, T) token ids, got {idx.shape}"
        B, T = idx.size()
        assert T <= self.config.max_length, (
            f"sequence length {T} exceeds max_length {self.config.max_length}"
        )
        assert idx.dtype in (torch.long, torch.int), (
            f"token ids must be integer dtype, got {idx.dtype}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        assert tok_emb.shape == (B, T, self.config.n_embed)
        assert pos_emb.shape == (T, self.config.n_embed)

        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        assert x.shape == (B, T, self.config.n_embed)
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, verbose: bool = True):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        self.gpt2_spec_init()

        if verbose:
            print("Number of parameters: ", sum(p.numel() for p in self.parameters()))

    def forward(self, idx):
        x = self.transformer(idx)
        logits = self.lm_head(x)
        assert logits.shape == (*idx.shape, self.config.vocab_size)
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gpt2_spec_init(self):
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, 16

    naive_cfg = get_config("tiny", attn_impl="naive", dropout=0.0)
    variants = {
        "sdpa": get_config("tiny", attn_impl="sdpa", dropout=0.0),
    }
    if torch.cuda.is_available():
        variants["triton"] = get_config(
            "tiny", attn_impl="triton", dropout=0.0, flash_block_size=32
        )
        variants["cuda"] = get_config(
            "tiny", attn_impl="cuda", dropout=0.0, flash_block_size=32
        )
        variants["numba"] = get_config(
            "tiny", attn_impl="numba", dropout=0.0, flash_block_size=32
        )

    naive_model = GPT(naive_cfg).eval()
    idx = torch.randint(0, naive_cfg.vocab_size, (B, T))
    device = torch.device("cuda" if "triton" in variants else "cpu")
    naive_model = naive_model.to(device)
    idx = idx.to(device)

    with torch.no_grad():
        out_naive = naive_model(idx)

    for name, cfg in variants.items():
        model = GPT(cfg, verbose=False).eval().to(device)
        model.load_state_dict(naive_model.state_dict())
        with torch.no_grad():
            out = model(idx)
        max_diff = (out_naive - out).abs().max().item()
        print(f"naive vs {name} max |diff| = {max_diff:.6e}")
        tol = 5e-3 if name in ("triton", "cuda", "numba") else 1e-3
        assert max_diff < tol, f"naive/{name} mismatch: {max_diff}"
    print("ok")
