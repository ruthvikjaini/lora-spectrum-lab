"""Tiny causal transformer with optional LoRA adapters (NumPy, manual backprop).

Single-head, one layer, d=64 by default.  Every linear layer can carry a
LoRA adapter: the effective weight is

    W_eff = W0 + (alpha / r) B A,

with W0 frozen during adapter training and only B, A updated.  With no
adapters attached the model is a plain full fine-tuning transformer (used
to pretrain the frozen base W0).

Backprop is hand-written and verified against finite differences in
tests/test_transformer_lora.py (all parameters, including LoRA factors).

Architecture (per position, causal):
    x   = E[tok] + P[pos]
    att = softmax(Q K^T / sqrt(d) + causal_mask) V,  Q,K,V = x @ Wq/Wk/Wv
    h   = x + att @ Wo
    mlp = relu(h @ W1 + b1) @ W2 + b2
    h2  = h + mlp
    y   = layernorm(h2) @ Wh + bh
"""

import numpy as np


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _layernorm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return xhat, mu, var


class Transformer:
    def __init__(self, vocab=15, d=64, hidden=128, max_len=32, seed=0,
                 lora_rank=0, lora_alpha=0.0, lora_layers=("Wq", "Wv", "Wo", "W1")):
        rng = np.random.default_rng(seed)
        self.vocab, self.d, self.hidden, self.max_len = vocab, d, hidden, max_len
        self.lora_rank, self.lora_alpha = lora_rank, lora_alpha
        self.lora_layers = tuple(lora_layers)
        s = 1.0 / np.sqrt(d)
        self.p = {
            "E": rng.standard_normal((vocab, d)) * s,
            "P": rng.standard_normal((max_len, d)) * s,
            "Wq": rng.standard_normal((d, d)) * s,
            "Wk": rng.standard_normal((d, d)) * s,
            "Wv": rng.standard_normal((d, d)) * s,
            "Wo": rng.standard_normal((d, d)) * s,
            "W1": rng.standard_normal((d, hidden)) * (1.0 / np.sqrt(d)),
            "b1": np.zeros(hidden),
            "W2": rng.standard_normal((hidden, d)) * (1.0 / np.sqrt(hidden)),
            "b2": np.zeros(d),
            "gamma": np.ones(d),
            "beta": np.zeros(d),
            "Wh": rng.standard_normal((d, vocab)) * s,
            "bh": np.zeros(vocab),
        }
        # LoRA factors, one (B, A) pair per adapted layer.  B0 = 0 and A0
        # Kaiming-uniform, as in the paper (Sec. 2.1).  B is (d_out, r),
        # A is (r, d_in).
        self.lora = {}
        if lora_rank > 0:
            for name in self.lora_layers:
                d_out, d_in = self.p[name].shape
                A0 = rng.uniform(-1.0 / np.sqrt(d_in), 1.0 / np.sqrt(d_in),
                                 size=(lora_rank, d_in))
                self.lora[name] = {
                    "B": np.zeros((d_out, lora_rank)),
                    "A": A0,
                }

    def eff(self, name):
        """Effective weight of a layer (base + LoRA increment if attached)."""
        W = self.p[name]
        if name in self.lora:
            fac = self.lora[name]
            W = W + (self.lora_alpha / self.lora_rank) * (fac["B"] @ fac["A"])
        return W

    def forward(self, tok):
        """tok: (B, T) int tokens. Returns (logits (B, T, V), cache)."""
        Bsz, T = tok.shape
        d = self.d
        p = self.p
        x = p["E"][tok] + p["P"][None, :T, :]
        Q = x @ self.eff("Wq")
        K = x @ self.eff("Wk")
        V = x @ self.eff("Wv")
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask[None, :, :], -1e9, scores)
        A = _softmax(scores)
        att = A @ V
        att_out = att @ self.eff("Wo")
        h = x + att_out
        m_in = h @ self.eff("W1") + p["b1"]
        m_act = np.maximum(m_in, 0.0)
        mlp = m_act @ self.eff("W2") + p["b2"]
        h2 = h + mlp
        xhat, mu, var = _layernorm(h2)
        ln = xhat * p["gamma"] + p["beta"]
        logits = ln @ p["Wh"] + p["bh"]
        cache = dict(tok=tok, x=x, Q=Q, K=K, V=V, A=A, att=att, att_out=att_out,
                     h=h, m_in=m_in, m_act=m_act, mlp=mlp, h2=h2, xhat=xhat,
                     mu=mu, var=var, ln=ln)
        return logits, cache

    def logprobs(self, tok):
        logits, _ = self.forward(tok)
        lp = logits - logits.max(axis=-1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(axis=-1, keepdims=True))
        return lp

    def backward(self, dlogits, cache):
        """dlogits: (B, T, V) gradient of the loss wrt logits.

        Returns grads: dict with the same keys as self.p, plus
        grads_lora[name] = {"B": ..., "A": ...} for adapted layers.
        Gradients wrt frozen base weights are computed (the caller decides
        which to apply); LoRA factor gradients use the chain rule through
        W_eff = W0 + (alpha/r) B A.
        """
        p, d = self.p, self.d
        Bsz, T = cache["tok"].shape
        x, Q, K, V, A = cache["x"], cache["Q"], cache["K"], cache["V"], cache["A"]
        att, h = cache["att"], cache["h"]
        m_in, m_act, h2 = cache["m_in"], cache["m_act"], cache["h2"]
        xhat, mu, var, ln = cache["xhat"], cache["mu"], cache["var"], cache["ln"]
        g = {}
        # Head.
        g["Wh"] = np.einsum("btd,btv->dv", ln, dlogits)
        g["bh"] = dlogits.sum(axis=(0, 1))
        dln = dlogits @ p["Wh"].T
        # LayerNorm: ln = xhat * gamma + beta.
        g["gamma"] = (dln * xhat).sum(axis=(0, 1))
        g["beta"] = dln.sum(axis=(0, 1))
        dxhat = dln * p["gamma"]
        # Layernorm backward.
        std = np.sqrt(var + 1e-5)
        dx = dxhat / std
        dvar = np.sum(dxhat * (h2 - mu) * (-0.5) * (var + 1e-5) ** (-1.5),
                      axis=-1, keepdims=True)
        dmu = np.sum(dxhat * (-1.0 / std), axis=-1, keepdims=True) + \
            dvar * np.mean(-2.0 * (h2 - mu), axis=-1, keepdims=True)
        dh2 = dx + dvar * 2.0 * (h2 - mu) / d + dmu / d
        # Residual: h2 = h + mlp.
        dh = dh2.copy()
        dmlp = dh2
        # MLP: mlp = relu(h @ W1 + b1) @ W2 + b2.
        W2 = self.eff("W2")
        g["W2"] = np.einsum("bth,btd->hd", m_act, dmlp)
        g["b2"] = dmlp.sum(axis=(0, 1))
        dm_act = dmlp @ W2.T
        dm_in = dm_act * (m_in > 0.0)
        W1 = self.eff("W1")
        g["W1"] = np.einsum("btd,bth->dh", h, dm_in)
        g["b1"] = dm_in.sum(axis=(0, 1))
        dh = dh + dm_in @ W1.T
        # Residual: h = x + att_out.
        dx_res = dh
        datt_out = dh
        Wo = self.eff("Wo")
        g["Wo"] = np.einsum("btd,bte->de", att, datt_out)
        datt = datt_out @ Wo.T
        # Attention: att = A @ V; A = softmax(scores).
        dV = A.transpose(0, 2, 1) @ datt
        dA = datt @ V.transpose(0, 2, 1)
        # Softmax backward (rows sum to 1).
        dscores = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
        # Masked positions had score -1e9; their gradient is ~0 already
        # (A ~ 0 there), but zero them explicitly for safety.
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dscores = np.where(mask[None, :, :], 0.0, dscores)
        dscores = dscores / np.sqrt(d)
        dQ = dscores @ K
        dK = dscores.transpose(0, 2, 1) @ Q
        Wq, Wk, Wv = self.eff("Wq"), self.eff("Wk"), self.eff("Wv")
        g["Wq"] = np.einsum("btd,bte->de", x, dQ)
        g["Wk"] = np.einsum("btd,bte->de", x, dK)
        g["Wv"] = np.einsum("btd,bte->de", x, dV)
        dx = dx_res + dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
        # Embedding + positional.
        gE = np.zeros_like(p["E"])
        np.add.at(gE, cache["tok"].reshape(-1),
                  dx.reshape(-1, d))
        g["E"] = gE
        gP = np.zeros_like(p["P"])
        gP[:T] = dx.sum(axis=0)
        g["P"] = gP
        # Split LoRA gradients out of the effective-weight gradients.
        glora = {}
        for name in self.lora:
            fac = self.lora[name]
            scale = self.lora_alpha / self.lora_rank
            gW = g[name]
            glora[name] = {
                "B": scale * (gW @ fac["A"].T),
                "A": scale * (fac["B"].T @ gW),
            }
        return g, glora

    def generate(self, prompt_tok, max_new, rng, temperature=1.0):
        """Greedy-or-sampled autoregressive generation.

        prompt_tok: (B, Tp) tokens. Returns (B, Tp + max_new) tokens.
        Sampling uses temperature; temperature 0 = greedy.
        """
        Bsz = prompt_tok.shape[0]
        seq = prompt_tok.copy()
        for _ in range(max_new):
            logits, _ = self.forward(seq)
            nxt = logits[:, -1, :]
            if temperature == 0.0:
                tok = np.argmax(nxt, axis=-1)
            else:
                z = nxt / temperature
                z = z - z.max(axis=-1, keepdims=True)
                pr = np.exp(z)
                pr = pr / pr.sum(axis=-1, keepdims=True)
                u = rng.random(Bsz)
                tok = (np.cumsum(pr, axis=-1) < u[:, None]).sum(axis=-1)
                tok = np.minimum(tok, self.vocab - 1)
            seq = np.concatenate([seq, tok[:, None]], axis=1)
        return seq
