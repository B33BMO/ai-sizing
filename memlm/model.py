"""Small GPT with an optional product-key memory layer.

Memory variants (cfg.mem_pointer / cfg.mem_hops):
  - flat:    one top-k read from a large key->value table (Lample'19 / Berges'24 style)
  - chained: hop h query = Wq_h(x) + Wp(read_{h-1})   (iterated retrieval)
  - pointer: every slot also owns a learned pointer vector into key space;
             hop 2 queries with the weighted pointers of hop-1 slots, so the
             memory is a soft, learned graph that a single forward pass can walk.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50304
    seq_len: int = 512
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    mlp_mult: float = 8 / 3
    # memory
    mem_layers: list = field(default_factory=list)  # block indices whose MLP is replaced by memory
    mem_n_keys: int = 512        # sub-keys per half -> n_keys**2 slots
    mem_heads: int = 4
    mem_topk: int = 32
    mem_k_dim: int = 256
    mem_hops: int = 1
    mem_pointer: str = "none"    # none | value | table
    # graph supervision of hop-2 (needs mem_hops >= 2): pointer reads at an entity mention
    # should land where hop-1 reads land for that entity's graph neighbours
    n_ent: int = 0               # 0 disables
    graph_tau: float = 0.1
    graph_samples: int = 512     # entity-mention positions used per forward


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.w)


def rope_cache(seq_len, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[: x.size(2)], sin[: x.size(2)]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hid = int(cfg.mlp_mult * cfg.n_embd)
        hid = 64 * ((hid + 63) // 64)
        self.up = nn.Linear(cfg.n_embd, 2 * hid, bias=False)
        self.down = nn.Linear(hid, cfg.n_embd, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class ProductKeyMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, H, dk = cfg.n_embd, cfg.mem_heads, cfg.mem_k_dim
        self.H, self.dk, self.k, self.nk = H, dk, cfg.mem_topk, cfg.mem_n_keys
        self.hops, self.pointer = cfg.mem_hops, cfg.mem_pointer
        n_slots = cfg.mem_n_keys ** 2
        self.n_slots = n_slots

        self.keys = nn.Parameter(torch.randn(H, 2, self.nk, dk // 2) / math.sqrt(dk // 2))
        self.values = nn.Embedding(n_slots, d)
        nn.init.normal_(self.values.weight, std=d ** -0.5)
        self.q_proj = nn.ModuleList([nn.Linear(d, H * dk, bias=False) for _ in range(self.hops)])
        if self.hops > 1:
            if self.pointer == "table":
                # every slot owns a pointer (per head) into key space: the learned graph edges
                self.pointers = nn.Embedding(n_slots, dk)
                nn.init.normal_(self.pointers.weight, std=0.02)
            else:  # "value" / "none": derive next query from what was read
                self.p_proj = nn.Linear(d, H * dk, bias=False)
                nn.init.zeros_(self.p_proj.weight)
        self.gate = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.graph = cfg.n_ent > 0 and self.hops > 1
        if self.graph:
            self.tau, self.g_samples = cfg.graph_tau, cfg.graph_samples
            # running mean of each entity's (normalized) hop-1 query: where its facts live in key space
            self.register_buffer("bank", torch.zeros(cfg.n_ent, H, dk))
            self.register_buffer("nbr", torch.full((cfg.n_ent, 32), -1, dtype=torch.long))

    def _graph_loss(self, q1, q2, ent):
        """InfoNCE over all entities: the hop-2 query at a mention of A should match the
        banked hop-1 queries of A's neighbours. q1, q2: (N, H, dk) pre-norm; ent: (N,) ids or -1."""
        N = ent.size(0)
        # fixed-size sample of labelled positions (static shapes for torch.compile)
        pick = (torch.rand(N, device=ent.device) + (ent >= 0).float() * 2).topk(min(self.g_samples, N)).indices
        e = ent[pick]
        valid = e >= 0
        e = e.clamp(min=0)
        q1 = F.layer_norm(q1[pick].float(), (self.dk,))
        q2 = F.layer_norm(q2[pick].float(), (self.dk,))
        with torch.no_grad():
            self.bank.index_add_(0, e, (q1 - self.bank[e]) * (0.05 * valid.float())[:, None, None])
        bank = F.normalize(self.bank, dim=-1)
        logits = torch.einsum("mhd,ehd->me", F.normalize(q2, dim=-1), bank) / (self.H * self.tau)
        nb = self.nbr[e]
        has = nb >= 0
        pos = logits.gather(1, nb.clamp(min=0)).masked_fill(~has, float("-inf"))
        use = valid & has.any(1)
        loss = torch.logsumexp(logits, 1) - torch.logsumexp(pos.masked_fill(~use[:, None], 0.0), 1)
        return torch.where(use, loss, torch.zeros_like(loss)).sum() / use.sum().clamp(min=1)

    def _lookup(self, q):
        # q: (N, H, dk) -> slot idx (N, H, k), weights (N, H, k)
        q = F.layer_norm(q.float(), (self.dk,))
        q1, q2 = q[..., : self.dk // 2], q[..., self.dk // 2:]
        k = self.keys.float()
        s1 = torch.einsum("nhd,hkd->nhk", q1, k[:, 0])
        s2 = torch.einsum("nhd,hkd->nhk", q2, k[:, 1])
        v1, i1 = s1.topk(self.k, dim=-1)
        v2, i2 = s2.topk(self.k, dim=-1)
        all_s = (v1[..., :, None] + v2[..., None, :]).flatten(-2)
        all_i = (i1[..., :, None] * self.nk + i2[..., None, :]).flatten(-2)
        s, j = all_s.topk(self.k, dim=-1)
        idx = all_i.gather(-1, j)
        return idx, F.softmax(s, dim=-1)

    @staticmethod
    def _bag(table, idx, w):
        # memory-efficient weighted gather-sum: (N, H, k) -> (N, H, D)
        N, H, k = idx.shape
        out = F.embedding_bag(idx.reshape(N * H, k), table, per_sample_weights=w.reshape(N * H, k).to(table.dtype), mode="sum")
        return out.view(N, H, -1)

    def forward(self, x, ent=None):
        B, T, C = x.shape
        xf = x.reshape(B * T, C)
        N = xf.size(0)
        total = 0
        q = self.q_proj[0](xf).view(N, self.H, self.dk)
        qs = [q]
        for h in range(self.hops):
            idx, w = self._lookup(q)
            read = self._bag(self.values.weight, idx, w).sum(1)  # (N, C)
            total = total + read
            if h + 1 < self.hops:
                q = self.q_proj[h + 1](xf).view(N, self.H, self.dk)
                if self.pointer == "table":
                    # follow the edges: weighted pointers of the slots each head just read
                    q = q + self._bag(self.pointers.weight, idx, w).to(q.dtype)  # (N, H, dk)
                else:
                    q = q + self.p_proj(read.to(xf.dtype)).view(N, self.H, self.dk)
                qs.append(q)
        y = total.to(x.dtype) * F.silu(self.gate(xf))
        aux = self._graph_loss(qs[0], qs[1], ent.reshape(-1)) if self.graph and ent is not None else None
        return self.out(y).view(B, T, C), aux


class Block(nn.Module):
    def __init__(self, cfg, use_mem):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ff = ProductKeyMemory(cfg) if use_mem else MLP(cfg)

    def forward(self, x, cos, sin, ent=None):
        x = x + self.attn(self.n1(x), cos, sin)
        if isinstance(self.ff, ProductKeyMemory):
            y, aux = self.ff(self.n2(x), ent)
            return x + y, aux
        return x + self.ff(self.n2(x)), None


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, i in cfg.mem_layers) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        cos, sin = rope_cache(cfg.seq_len, cfg.n_embd // cfg.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for name, p in self.named_parameters():
            if p.dim() == 2 and "values" not in name and "pointers" not in name and "p_proj" not in name:
                nn.init.normal_(p, std=0.02)
            if name.endswith(("proj.weight", "down.weight", "out.weight")) and "q_proj" not in name and "p_proj" not in name:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, targets=None, ent=None):
        """-> logits, or (logits, lm_loss, graph_loss) when targets are given."""
        x = self.wte(idx)
        aux = torch.zeros((), device=idx.device)
        for b in self.blocks:
            x, a = b(x, self.cos, self.sin, ent)
            if a is not None:
                aux = aux + a
        logits = self.lm_head(self.norm(x)).float()
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1), aux

    def set_graph(self, nbr):
        for b in self.blocks:
            if isinstance(b.ff, ProductKeyMemory) and b.ff.graph:
                b.ff.nbr.copy_(torch.as_tensor(nbr, dtype=torch.long))

    def param_counts(self):
        mem = sum(p.numel() for n, p in self.named_parameters() if ".ff.values" in n or ".ff.pointers" in n)
        emb = self.wte.weight.numel()
        tot = sum(p.numel() for p in self.parameters())
        return {"total": tot, "embedding": emb, "memory_table": mem, "core": tot - emb - mem}
