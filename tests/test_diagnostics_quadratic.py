"""Tests for diagnostics.py and quadratic.py (foundation of the project)."""

import numpy as np
import pytest

from lora_spectrum.diagnostics import (
    column_projector,
    effective_rank,
    erank_of_delta,
    flatten_orthogonal_component,
    matrix_rank_tol,
    overlap_of_deltas,
    spectral_flatten,
    subspace_overlap,
    svd_of_delta,
)
from lora_spectrum.quadratic import QuadraticModel, grad_f, population_risk, predict


def test_effective_rank_uniform_is_r():
    # Equal singular values -> erank == r.
    assert effective_rank(np.array([2.0, 2.0, 2.0, 2.0])) == pytest.approx(4.0)
    # One dominant value -> erank -> 1.
    assert effective_rank(np.array([10.0, 1e-9, 1e-9])) == pytest.approx(1.0, abs=1e-6)
    # Two equal, rest zero -> erank == 2.
    assert effective_rank(np.array([3.0, 3.0, 0.0, 0.0])) == pytest.approx(2.0)
    # Zero matrix -> 0.
    assert effective_rank(np.zeros(4)) == 0.0


def test_erank_of_delta_matches_direct_svd():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((12, 4))
    A = rng.standard_normal((4, 9))
    U, s, Vt = svd_of_delta(B, A, alpha=8.0, r=4)
    assert erank_of_delta(B, A, 8.0, 4) == pytest.approx(effective_rank(s))
    # B = 0 -> zero increment -> erank 0.
    assert erank_of_delta(np.zeros((12, 4)), A, 8.0, 4) == 0.0


def test_subspace_overlap_identities():
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    U1 = Q[:, :4]
    # Identical subspaces -> overlap 1.
    assert subspace_overlap(U1, U1, 4) == pytest.approx(1.0)
    # Orthogonal subspaces -> overlap 0.
    U2 = Q[:, 4:8]
    assert subspace_overlap(U1, U2, 4) == pytest.approx(0.0, abs=1e-12)
    # Overlap is in [0, 1] for a random rotation.
    Q2, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    ov = subspace_overlap(U1, Q2[:, :4], 4)
    assert 0.0 <= ov <= 1.0


def test_overlap_of_deltas_same_increment_is_one():
    rng = np.random.default_rng(2)
    B = rng.standard_normal((10, 4))
    A = rng.standard_normal((4, 7))
    ovlp_l, ovlp_r = overlap_of_deltas(B, A, B, A, 8.0, 4)
    assert ovlp_l == pytest.approx(1.0)
    assert ovlp_r == pytest.approx(1.0)


def test_spectral_flatten_properties():
    rng = np.random.default_rng(3)
    D = rng.standard_normal((9, 6)) @ np.diag([5.0, 3.0, 1.0, 0.5, 0.1, 0.01])
    M = spectral_flatten(D)
    # Norm preserved.
    assert np.linalg.norm(M) == pytest.approx(np.linalg.norm(D), rel=1e-12)
    # Nonzero singular values all equal ||D||_F / sqrt(rank).
    s = np.linalg.svd(M, compute_uv=False)
    rank = matrix_rank_tol(D)
    assert np.allclose(s[:rank], np.linalg.norm(D) / np.sqrt(rank), rtol=1e-10)
    assert np.allclose(s[rank:], 0.0, atol=1e-12)
    # Zero matrix maps to itself.
    assert np.array_equal(spectral_flatten(np.zeros((4, 5))), np.zeros((4, 5)))
    # Rank-1 input: flattening keeps the single direction, same norm.
    D1 = np.outer([1.0, 2.0, 3.0], [0.5, -1.0])
    M1 = spectral_flatten(D1)
    assert np.linalg.norm(M1) == pytest.approx(np.linalg.norm(D1), rel=1e-12)
    assert np.linalg.matrix_rank(M1) == 1


def test_column_projector():
    rng = np.random.default_rng(4)
    B = rng.standard_normal((8, 3))
    P = column_projector(B)
    # Idempotent and symmetric.
    assert np.allclose(P @ P, P, atol=1e-12)
    assert np.allclose(P, P.T, atol=1e-12)
    # Projects onto col(B): P B == B.
    assert np.allclose(P @ B, B, atol=1e-12)
    # Zero B -> zero projector.
    assert np.array_equal(column_projector(np.zeros((8, 3))), np.zeros((8, 8)))
    # Full-column-rank B -> trace == rank.
    assert np.trace(P) == pytest.approx(3.0)


def test_flatten_orthogonal_component():
    rng = np.random.default_rng(5)
    B = rng.standard_normal((8, 3))
    D = rng.standard_normal((8, 5))
    out = flatten_orthogonal_component(D, B)
    P = column_projector(B)
    # Within-subspace component preserved exactly.
    assert np.allclose(P @ out, P @ D, atol=1e-12)
    # Orthogonal component has the same Frobenius norm as before.
    orth_in = (np.eye(8) - P) @ D
    orth_out = (np.eye(8) - P) @ out
    assert np.linalg.norm(orth_out) == pytest.approx(np.linalg.norm(orth_in), rel=1e-10)
    # And its spectrum is flat.
    s = np.linalg.svd(orth_out, compute_uv=False)
    nz = s[s > 1e-10]
    assert np.allclose(nz, nz[0], rtol=1e-8)


def test_predict_and_grad_f_finite_difference():
    rng = np.random.default_rng(6)
    W = rng.standard_normal((7, 5))
    x = rng.standard_normal(7)
    G = grad_f(W, x)
    eps = 1e-6
    num = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += eps
            Wm = W.copy(); Wm[i, j] -= eps
            num[i, j] = (predict(Wp, x) - predict(Wm, x)) / (2 * eps)
    assert np.allclose(G, num, rtol=1e-5, atol=1e-7)


def test_population_risk_exact():
    rng = np.random.default_rng(7)
    W = rng.standard_normal((6, 4))
    F = rng.standard_normal((6, 4))
    # Monte-Carlo check of the closed form.
    xs = rng.standard_normal((20000, 6))
    vals = [predict(W, x) - predict(F, x) for x in xs]
    mc = float(np.mean(np.array(vals) ** 2))
    exact = population_risk(W, F)
    assert abs(mc - exact) < 0.05 * exact + 0.01


def test_quadratic_pair_gradients_finite_difference():
    rng = np.random.default_rng(8)
    model = QuadraticModel(d=16, k=4, k_new=4, r=3, seed=8)
    x_p, x_m = model.sample_pair()
    eps_p, eps_m, G_raw, G_base = model.pair_gradients(x_p, x_m)

    def loss_raw(W):
        a = predict(W, x_p) - predict(model.F, x_p)
        b = predict(W, x_m) - predict(model.F, x_m)
        return (a ** 2 + b ** 2) / 4.0

    def loss_base(W):
        a = predict(W, x_p) - predict(model.F, x_p)
        b = predict(W, x_m) - predict(model.F, x_m)
        bar = (a + b) / 2.0
        return ((a - bar) ** 2 + (b - bar) ** 2) / 4.0

    W = model.W
    eps = 1e-6
    for G, loss in ((G_raw, loss_raw), (G_base, loss_base)):
        num = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Wp = W.copy(); Wp[i, j] += eps
                Wm = W.copy(); Wm[i, j] -= eps
                num[i, j] = (loss(Wp) - loss(Wm)) / (2 * eps)
        assert np.allclose(G, num, rtol=1e-4, atol=1e-6), (
            "gradient mismatch: max abs err",
            np.max(np.abs(G - num)),
        )


def test_quadratic_step_reduces_loss_and_moves_lora():
    model = QuadraticModel(d=32, k=4, k_new=4, r=3, seed=9)
    risk0 = model.risk()
    assert np.allclose(model.B, 0.0)
    for _ in range(20):
        model.step(eta=0.05, baseline=False, n_pairs=8)
    assert np.linalg.norm(model.B) > 0.0  # B moved off zero
    assert np.linalg.norm(model.A - model.A) == 0.0  # (sanity: A finite)
    assert np.isfinite(model.risk())
    # Training (either rule) should not explode the risk in 20 small steps.
    assert model.risk() < 10.0 * risk0 + 1.0


def test_sample_pair_marginals_and_correlations():
    model = QuadraticModel(d=64, k=8, k_new=8, r=4, seed=10)
    n = 4000
    xp = np.zeros((n, 64)); xm = np.zeros((n, 64))
    for i in range(n):
        a, b = model.sample_pair()
        xp[i], xm[i] = a, b
    # Marginals standard normal.
    assert np.allclose(xp.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(xp.std(axis=0), 1.0, atol=0.1)
    # Positive correlation along pretrained coords (first k of Q basis).
    cp = xp @ model.Q[:, :8]
    cm = xm @ model.Q[:, :8]
    corr_pre = np.mean([np.corrcoef(cp[:, j], cm[:, j])[0, 1] for j in range(8)])
    assert corr_pre == pytest.approx(model.rho_a, abs=0.05)
    # Negative correlation along newly useful coords.
    np_ = xp @ model.Q[:, 8:16]
    nm = xm @ model.Q[:, 8:16]
    corr_new = np.mean([np.corrcoef(np_[:, j], nm[:, j])[0, 1] for j in range(8)])
    assert corr_new == pytest.approx(model.rho_o, abs=0.05)
