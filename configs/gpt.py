import json
from dataclasses import asdict, dataclass
from pathlib import Path

from attention import ATTN_IMPLS

CONFIG_DIR = Path(__file__).resolve().parent


@dataclass
class GPTConfig:
    vocab_size: int = 1000
    n_embed: int = 128
    n_heads: int = 4
    n_layers: int = 4
    n_intermediate: int = 4
    max_length: int = 128
    dropout: float = 0.0
    attn_impl: str = "naive"
    flash_block_size: int = 32

    def __post_init__(self):
        assert self.vocab_size > 0
        assert self.n_embed > 0
        assert self.n_heads > 0
        assert self.n_embed % self.n_heads == 0, (
            f"n_embed ({self.n_embed}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_layers > 0
        assert self.n_intermediate > 0
        assert self.max_length > 0
        assert 0.0 <= self.dropout < 1.0
        assert self.attn_impl in ATTN_IMPLS, (
            f"attn_impl must be one of {ATTN_IMPLS}, got {self.attn_impl!r}"
        )
        assert self.flash_block_size > 0
        if self.attn_impl in ("triton", "cuda", "numba") and self.dropout > 0.0:
            raise AssertionError(
                f"{self.attn_impl} attn_impl does not support dropout; set dropout=0.0"
            )

    def with_updates(self, **kwargs) -> "GPTConfig":
        data = asdict(self)
        data.update(kwargs)
        return GPTConfig(**data)


def list_configs():
    names = {p.stem for p in CONFIG_DIR.glob("*.json")}
    return sorted(names)


def get_config(name: str, **overrides) -> GPTConfig:
    path = CONFIG_DIR / f"{name}.json"
    assert path.is_file(), f"unknown config {name!r}; choose from {list_configs()}"
    with path.open() as f:
        data = json.load(f)
    cfg = GPTConfig(**data)
    if overrides:
        cfg = cfg.with_updates(**overrides)
    return cfg
