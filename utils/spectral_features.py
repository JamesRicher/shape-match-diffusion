"""Spectral point descriptors from the Laplace-Beltrami eigen-decomposition.

These are intrinsic (isometry-invariant) and directly comparable across shapes, computed from
the cached LBO spectrum (evals, evecs) that the datasets already load under ret_evecs. Used as a
non-learned, network-free feature source for the densifier's data term.
"""
import torch


def wks(evals: torch.Tensor, evecs: torch.Tensor, n_e: int = 100,
        variance: float = 7.0, eps: float = 1e-8) -> torch.Tensor:
    """Wave Kernel Signature per vertex (Aubry et al. 2011), auto energy range.

    Follows the standard auto-WKS: n_e energy samples spread over the log-eigenvalue range with a
    Gaussian band of width sigma = variance * (e_max - e_min) / n_e, the ends trimmed by 2 sigma.

    Args:
        evals: (K,) LBO eigenvalues, ascending and non-negative (the leading near-zero mode is
            fine -- its log falls far below the energy range and contributes negligibly).
        evecs: (N, K) LBO eigenvectors aligned to evals.
        n_e: number of energy samples = descriptor dimension.
        variance: energy-band width factor (7 in the reference implementation).

    Returns (N, n_e) descriptors.
    """
    log_ev = torch.log(evals.abs().clamp_min(eps))              # (K,)
    e_min, e_max = log_ev[1], log_ev[-1]                        # skip the ~0 leading eigenvalue
    sigma = variance * (e_max - e_min) / n_e
    energies = torch.linspace(e_min + 2 * sigma, e_max - 2 * sigma, n_e, device=evals.device)
    # coefs[k, i] = exp(-(e_i - log lambda_k)^2 / 2 sigma^2)
    coefs = torch.exp(-(energies[None, :] - log_ev[:, None]) ** 2 / (2 * sigma ** 2 + eps))  # (K, n_e)
    desc = (evecs ** 2) @ coefs                                 # (N, n_e)
    return desc / coefs.sum(0, keepdim=True).clamp_min(eps)     # per-energy normalisation
