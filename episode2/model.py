import torch
import torch.nn as nn


def topk_sample(probs: torch.Tensor, k: int) -> torch.Tensor:
    topk_probs, topk_inds = torch.topk(probs, k, dim=-1)
    sampled_inds = torch.multinomial(topk_probs, 1)
    output_inds = topk_inds.gather(-1, sampled_inds)
    return output_inds


def topp_sample(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_inds = probs.sort(-1, descending=True)
    sorted_cdf = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    sorted_probs.masked_fill_(sorted_cdf >= p, 0)
    sampled_inds = torch.multinomial(sorted_probs, 1)
    output_inds = sorted_inds.gather(-1, sampled_inds)
    return output_inds


def sample(probs: torch.Tensor, top_p: float = 1.0, top_k: int = -1) -> torch.Tensor:
    if top_p < 1:
        return topp_sample(probs, top_p)
    if top_k > 0:
        return topk_sample(probs, top_k)
    return torch.multinomial(probs, 1)


class MLP(nn.Module):
    def __init__(self, model_dims: int, mlp_factor: int = 4):
        super().__init__()
        self.h = nn.Linear(model_dims, model_dims * mlp_factor)
        self.o = nn.Linear(model_dims * mlp_factor, model_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.gelu(self.h(x))
        o = self.o(z)
        return o


class QKVAttention(nn.Module):
    def __init__(self, model_dims: int, n_heads: int):
        super().__init__()
        self.shape = n_heads, model_dims // n_heads
        self.qkv = nn.Linear(model_dims, 3 * model_dims)
        self.o = nn.Linear(model_dims, model_dims)

    def forward(self, x: torch.Tensor, cache: dict = None) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (x.unflatten(-1, self.shape).movedim(-3, -2) for x in (q, k, v))
        is_causal = True
        if cache is not None:
            if "k" in cache and "v" in cache:
                k = torch.cat([cache["k"], k], dim=-2)
                v = torch.cat([cache["v"], v], dim=-2)
                is_causal = False
            cache["k"] = k
            cache["v"] = v
        qkv = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        o = self.o(qkv.movedim(-3, -2).flatten(start_dim=2))
        return o


class TransformerLayer(nn.Module):
    def __init__(self, model_dims: int, n_heads: int, mlp_factor: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dims)
        self.qkv = QKVAttention(model_dims, n_heads)
        self.ln2 = nn.LayerNorm(model_dims)
        self.mlp = MLP(model_dims, mlp_factor)
        self.qkv.o.is_residual = True
        self.mlp.o.is_residual = True

    def forward(self, x: torch.Tensor, cache: dict = None) -> torch.Tensor:
        x = x + self.qkv(self.ln1(x), cache=cache)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerBlock(nn.ModuleList):
    def __init__(self, n_layers: int, model_dims: int, n_heads: int, mlp_factor: int = 4):
        super().__init__()
        for _ in range(n_layers):
            self.append(TransformerLayer(model_dims, n_heads, mlp_factor))
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            n_layers = len(self)
            if hasattr(module, "is_residual") and module.is_residual:
                std *= (2 * n_layers) ** -0.5
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, cache: dict = None) -> torch.Tensor:
        for i, layer in enumerate(self):
            if cache is not None:
                if f"layer{i}" not in cache:
                    cache[f"layer{i}"] = {}
                cache_i = cache[f"layer{i}"]
            else:
                cache_i = None
            x = layer(x, cache=cache_i)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50_304,
        max_seq_length: int = 1024,
        model_dims: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        mlp_factor: int = 4,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, model_dims)
        self.position_embeddings = nn.Embedding(max_seq_length, model_dims)
        self.body = TransformerBlock(n_layers=n_layers, model_dims=model_dims, n_heads=n_heads, mlp_factor=mlp_factor)
        self.ln = nn.LayerNorm(model_dims)
        self.head = nn.Linear(model_dims, vocab_size, bias=False)
        self.head.weight = self.token_embeddings.weight
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, cache: dict = None) -> torch.Tensor:
        seq_length = x.shape[1]
        if cache is not None:
            if "offset" not in cache:
                cache["offset"] = 0
            offset = cache["offset"]
        else:
            offset = 0
        pos = torch.arange(offset, offset + seq_length, dtype=torch.long, device=x.device)
        e = self.token_embeddings(x) + self.position_embeddings(pos)
        z = self.ln(self.body(e, cache=cache))
        yhat = self.head(z)
        if cache is not None:
            cache["offset"] += seq_length
        if y is None:
            return yhat
        loss = torch.nn.functional.cross_entropy(yhat.flatten(end_dim=-2), y.flatten())
        return yhat, loss

    def sample(
        self, inputs: torch.Tensor, max_length: int = 30, top_p: float = 1.0, top_k: int = -1, use_cache: bool = False
    ) -> torch.Tensor:
        cache = {} if use_cache else None
        outputs = []
        with torch.inference_mode():
            for _ in range(max_length):
                logits = self(inputs, cache=cache)[:, -1]
                samples = sample(logits.softmax(dim=-1), top_p=top_p, top_k=top_k)
                if use_cache:
                    inputs = samples
                else:
                    inputs = torch.cat([inputs, samples], dim=-1)
                outputs.append(samples)
        return torch.cat(outputs, dim=-1)
