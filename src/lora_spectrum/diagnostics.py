"""Spectral and subspace diagnostics for LoRA increments.

Implements the diagnostics of Liu et al., "Spectral and Subspace Dynamics of
LoRA in Supervised Fine-Tuning and Reinforcement Learning"
(arXiv:2611.08414, 5 Nov 2026), Section 2.1, and the norm-preserving
spectral flattening of Section 4.1.

For a LoRA increment  dW = (alpha / r) B A  with singular values
sigma_1 >= ... >= sigma_r >= 0:

* effective rank (Roy & Vetterli):  erank = exp(-sum_i p_i log p_i)
  with p_i = sigma_i^2 / sum_j sigma_j^2.  erank in [1, r]; smaller
  values mean stronger concentration in a few dominant directions.
* left/right subspace overlap between steps t1 and t2:
  ovlp_L = (1/r) ||U_t1^T U_t2||_F^2,  ovlp_R = (1/r) ||V_t1^T V_t2||_F^2.
  Smaller overlap over the same interval means faster rotation.
* norm-preserving spectral flattening (Muon variant, paper Eq. 4.1):
  M(D) = (||D||_F / sqrt(rank D)) D [(D^T D)^+]^{1/2},
  which replaces every nonzero singular value of D by
  ||D||_F / sqrt(rank D) while preserving ||M(D)||_F = ||D||_F.

Everything here is exact NumPy linear algebra; no model code is imported.
"""

import numpy as np


def svd_of_delta(B, A, alpha, r):
    """SVD of the LoRA increment dW = (alpha / r) B A.

    Returns (U, sigma, Vt) with sigma sorted descending.  B is (d_o, r),
    A is (r, d_i).
    """
    dW = (alpha / r) * (B @ A)
    U, sigma, Vt = np.linalg.svd(dW, full_matrices=False)
    return U, sigma, Vt


def effective_rank(sigma):
    """Effective rank from a vector of singular values (descending)."""
    sigma = np.asarray(sigma, dtype=np.float64)
    sq = sigma ** 2
    total = sq.sum()
    if total <= 0.0:
        return 0.0
    p = sq / total
    p = p[p > 0.0]
    return float(np.exp(-np.sum(p * np.log(p))))


def erank_of_delta(B, A, alpha, r):
    """Effective rank of the LoRA increment (alpha / r) B A."""
    _, sigma, _ = svd_of_delta(B, A, alpha, r)
    return effective_rank(sigma)


def subspace_overlap(U1, U2, r):
    """(1/r) ||U1^T U2||_F^2 for orthonormal bases U1, U2 (each (d, r))."""
    return float(np.linalg.norm(U1.T @ U2, ord="fro") ** 2) / float(r)


def overlap_of_deltas(B1, A1, B2, A2, alpha, r):
    """Left/right subspace overlaps between two LoRA increments."""
    U1, _, Vt1 = svd_of_delta(B1, A1, alpha, r)
    U2, _, Vt2 = svd_of_delta(B2, A2, alpha, r)
    # dW = (alpha/r) B A has rank <= r, so only the first r singular
    # vectors span the increment's subspaces; the rest of the SVD is an
    # arbitrary orthonormal completion and must not enter the overlap.
    ovlp_l = subspace_overlap(U1[:, :r], U2[:, :r], r)
    ovlp_r = subspace_overlap(Vt1.T[:, :r], Vt2.T[:, :r], r)
    return ovlp_l, ovlp_r


def matrix_rank_tol(D, tol=None):
    """Numerical rank of D: number of singular values above tolerance."""
    s = np.linalg.svd(D, compute_uv=False)
    if tol is None:
        tol = max(D.shape) * np.finfo(np.float64).eps * (s[0] if len(s) else 0.0)
    return int(np.sum(s > tol))


def spectral_flatten(D):
    """Norm-preserving spectral flattening M(D) (paper Sec. 4.1).

    M(D) = (||D||_F / sqrt(rank D)) * D @ [(D^T D)^+]^{1/2}.
    In SVD terms D = U diag(s) V^T, and M(D) = c * U V^T with
    c = ||D||_F / sqrt(rank D) on the nonzero singular subspace.
    Returns M(D) with ||M(D)||_F == ||D||_F and a flat nonzero spectrum.
    """
    D = np.asarray(D, dtype=np.float64)
    nrm = np.linalg.norm(D, ord="fro")
    if nrm == 0.0:
        return D.copy()
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    # Numerical rank: singular values above the standard tolerance.
    # (A plain `s > 0` test counts ~1e-17 numerical zeros as rank.)
    tol = max(D.shape) * np.finfo(np.float64).eps * s[0]
    rank = int(np.sum(s > tol))
    if rank == 0:
        return D.copy()
    c = nrm / np.sqrt(rank)
    # U[:, :rank] @ Vt[:rank, :] is the partial isometry U V^T.
    return c * (U[:, :rank] @ Vt[:rank, :])


def column_projector(B):
    """Orthogonal projector P onto the column space of B.

    P = B (B^T B)^+ B^T.  Computed as U U^T from the SVD of B, which is
    exact and stable.  Returns the zero matrix when B is (numerically) 0.
    """
    B = np.asarray(B, dtype=np.float64)
    if B.size == 0 or np.linalg.norm(B) == 0.0:
        return np.zeros((B.shape[0], B.shape[0]))
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    tol = max(B.shape) * np.finfo(np.float64).eps * (s[0] if len(s) else 0.0)
    rank = int(np.sum(s > tol))
    if rank == 0:
        return np.zeros((B.shape[0], B.shape[0]))
    U_r = U[:, :rank]
    return U_r @ U_r.T


def flatten_orthogonal_component(D, B):
    """Paper Sec. 4.1 RL update for B: P D + M((I - P) D).

    P is the projector onto the column space of the current B.  The
    within-subspace component is preserved; only the orthogonal component
    is spectrally flattened (its Frobenius norm is unchanged by M).
    """
    P = column_projector(B)
    within = P @ D
    orth = D - within
    return within + spectral_flatten(orth)
