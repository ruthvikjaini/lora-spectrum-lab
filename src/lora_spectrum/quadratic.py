"""Quadratic multi-index model of Liu et al. (arXiv:2611.08414), Section 3.

The paper's minimal model for isolating the effect of group-mean baseline
subtraction on LoRA dynamics.  Prediction:

    f_W(x) = (||W^T x||^2 - ||W||_F^2) / sqrt(2),   W = W0 + B A.

Teacher labels come from F_delta = U0 Sigma0 + delta * Z, where
W0 = U0 Sigma0 V0^T is the compact SVD of the pretrained weight and Z is a
random task perturbation with components along pretrained directions and an
orthogonal set of newly useful directions.  delta controls the size of the
post-training change.

Training data are pairs (x_+, x_-) of correlated, marginally standard
Gaussian inputs -- the smallest group containing both shared context and
within-group variation.  Positive correlation rho_a along pretrained
directions models the common prompt; negative correlation rho_o along newly
useful directions models completion-specific contrasts; the remaining q
task-irrelevant coordinates vary independently.

Two training rules, identical except for the baseline:
  without baseline:  L = (eps_+^2 + eps_-^2) / 4
  with baseline:     L = ((eps_+ - eps_bar)^2 + (eps_- - eps_bar)^2) / 4
where eps_+- = f_W(x_+-) - f_F(x_+-) and eps_bar = (eps_+ + eps_-)/2.
Both rules are evaluated on the same ordinary population prediction risk,
which for x ~ N(0, I) is exactly ||W W^T - F F^T||_F^2 (derived in the
module docstring of risk()).

LoRA factors are trained by SGD with unit scaling (alpha / r = 1), B0 = 0,
A0 with independent Gaussian entries, exactly as in the paper.
"""

import numpy as np

SQRT2 = float(np.sqrt(2.0))


def predict(W, x):
    """f_W(x) = (||W^T x||^2 - ||W||_F^2) / sqrt(2)."""
    u = W.T @ x
    return float((u @ u - np.sum(W * W)) / SQRT2)


def grad_f(W, x):
    """d f_W(x) / dW = sqrt(2) * (x u^T - W), u = W^T x."""
    u = W.T @ x
    return SQRT2 * (np.outer(x, u) - W)


def population_risk(W, F):
    """Exact population risk E_x[(f_W(x) - f_F(x))^2] for x ~ N(0, I).

    f_W - f_F = (x^T M x - tr M) / sqrt(2) with M = W W^T - F F^T
    symmetric.  For standard Gaussian x, Var(x^T M x) = 2 ||M||_F^2,
    so the risk is ||W W^T - F F^T||_F^2.
    """
    M = W @ W.T - F @ F.T
    return float(np.sum(M * M))


class QuadraticModel:
    """Quadratic multi-index model with a LoRA adapter trained by SGD."""

    def __init__(self, d=512, k=8, k_new=8, r=4, delta=0.05,
                 rho_a=0.7, rho_o=-0.5, seed=0):
        self.d, self.k, self.k_new, self.r = d, k, k_new, r
        self.q = d - k - k_new
        assert self.q > 0, "need q = d - k - k_new > 0 task-irrelevant coords"
        self.delta = delta
        self.rho_a, self.rho_o = rho_a, rho_o
        rng = np.random.default_rng(seed)
        self.rng = rng
        # Random orthogonal basis Q: first k coords = pretrained directions,
        # next k_new = newly useful directions, rest = task-irrelevant.
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        self.Q = Q
        # Pretrained weight W0 = U0 Sigma0 V0^T, shape (d, k).
        U0 = Q[:, :k]
        sing = np.linspace(2.0, 0.5, k)  # separated feature strengths
        V0, _ = np.linalg.qr(rng.standard_normal((k, k)))
        self.W0 = (U0 * sing) @ V0.T
        # Teacher F_delta = U0 Sigma0 + delta * Z (paper Sec. 3, literal
        # form).  At delta = 0 the teacher and the pretrained student make
        # identical predictions, because f_W depends on W only through
        # W W^T and ||W||_F^2, and (U0 Sigma0)(U0 Sigma0)^T = W0 W0^T.
        # Z has components along pretrained directions and along the newly
        # useful directions (the next k_new columns of Q).
        Z_pre = rng.standard_normal((k, k)) * 0.3
        Z_new = rng.standard_normal((k_new, k))
        Z = np.zeros((d, k))
        Z[:k, :] = Z_pre
        Z[k:k + k_new, :] = Z_new
        Z = Q @ Z
        self.F = (U0 * sing) + delta * Z
        # LoRA factors: B0 = 0, A0 iid Gaussian (paper Sec. 3).
        self.B = np.zeros((d, r))
        self.A = rng.standard_normal((r, k)) / np.sqrt(k)

    @property
    def W(self):
        # Unit LoRA scaling in the paper's Section 3 model.
        return self.W0 + self.B @ self.A

    def sample_pair(self):
        """One correlated input pair (x_+, x_-), marginally N(0, I_d)."""
        rng = self.rng
        k, k_new, q = self.k, self.k_new, self.q
        coords_p = np.zeros(self.d)
        coords_m = np.zeros(self.d)
        # Pretrained coords: positive correlation rho_a (shared prompt).
        g_a = rng.standard_normal(k)
        coords_p[:k] = (np.sqrt(self.rho_a) * g_a
                        + np.sqrt(1 - self.rho_a) * rng.standard_normal(k))
        coords_m[:k] = (np.sqrt(self.rho_a) * g_a
                        + np.sqrt(1 - self.rho_a) * rng.standard_normal(k))
        # Newly useful coords: negative correlation rho_o (contrasts).
        g_o = rng.standard_normal(k_new)
        s = np.sqrt(abs(self.rho_o))
        coords_p[k:k + k_new] = (s * g_o
                                 + np.sqrt(1 - abs(self.rho_o))
                                 * rng.standard_normal(k_new))
        coords_m[k:k + k_new] = (-s * g_o
                                 + np.sqrt(1 - abs(self.rho_o))
                                 * rng.standard_normal(k_new))
        # Task-irrelevant coords: independent.
        coords_p[k + k_new:] = rng.standard_normal(q)
        coords_m[k + k_new:] = rng.standard_normal(q)
        return self.Q @ coords_p, self.Q @ coords_m

    def pair_gradients(self, x_p, x_m):
        """Residuals and dL/dW for one pair, with and without baseline.

        Returns (eps_p, eps_m, G_raw, G_base) where G_raw is the gradient
        of (eps_p^2 + eps_m^2)/4 and G_base of the baseline-subtracted
        loss ((eps_p - eps_bar)^2 + (eps_m - eps_bar)^2)/4.
        """
        W = self.W
        eps_p = predict(W, x_p) - predict(self.F, x_p)
        eps_m = predict(W, x_m) - predict(self.F, x_m)
        Gf_p = grad_f(W, x_p)
        Gf_m = grad_f(W, x_m)
        G_raw = 0.5 * (eps_p * Gf_p + eps_m * Gf_m)
        diff = eps_p - eps_m
        # Baseline loss = (eps_p - eps_m)^2 / 8; d/d eps_p = diff/4,
        # d/d eps_m = -diff/4.  The -W terms cancel between the two terms.
        G_base = 0.25 * diff * (Gf_p - Gf_m)
        return eps_p, eps_m, G_raw, G_base

    def step(self, eta, baseline, n_pairs=8):
        """One SGD step on n_pairs freshly sampled pairs; returns mean loss."""
        dB = np.zeros_like(self.B)
        dA = np.zeros_like(self.A)
        loss = 0.0
        for _ in range(n_pairs):
            x_p, x_m = self.sample_pair()
            eps_p, eps_m, G_raw, G_base = self.pair_gradients(x_p, x_m)
            G = G_base if baseline else G_raw
            # Unit scaling: dL/dB = G A^T, dL/dA = B^T G.
            dB += G @ self.A.T
            dA += self.B.T @ G
            if baseline:
                loss += (eps_p - eps_m) ** 2 / 8.0
            else:
                loss += (eps_p ** 2 + eps_m ** 2) / 4.0
        dB /= n_pairs
        dA /= n_pairs
        self.B -= eta * dB
        self.A -= eta * dA
        return loss / n_pairs

    def risk(self):
        return population_risk(self.W, self.F)
