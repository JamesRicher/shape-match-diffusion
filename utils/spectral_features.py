"""Spectral point descriptors from the Laplace-Beltrami eigen-decomposition.

These are intrinsic (isometry-invariant) and directly comparable across shapes, computed from
the cached LBO spectrum (evals, evecs) that the datasets already load under ret_evecs. Used as a
non-learned, network-free feature source for the densifier data terms.
"""
import torch


def wks_grid(evals: torch.Tensor, n_e: int = 100, variance: float = 7.0, eps: float = 1e-8):
    """Standard auto-WKS energy grid for one shape: n_e log-eigenvalue samples spanning
    [log lambda_min, log lambda_max] (leading ~0 mode skipped), band width sigma, ends trimmed
    by 2 sigma. Returns (energies (n_e,), sigma)."""
    log_ev = torch.log(evals.abs().clamp_min(eps))
    e_min, e_max = log_ev[1], log_ev[-1]                        # skip the ~0 leading eigenvalue
    sigma = variance * (e_max - e_min) / n_e
    energies = torch.linspace(e_min + 2 * sigma, e_max - 2 * sigma, n_e, device=evals.device)
    return energies, sigma


def shared_wks_grid(evals_x: torch.Tensor, evals_y: torch.Tensor, n_e: int = 100,
                    variance: float = 7.0, eps: float = 1e-8):
    """One energy grid covering BOTH spectra, so band i means the same energy on each shape (the
    prerequisite for functional-map descriptor correspondence). Range is the overlap of the two
    log-eigenvalue spans -- [max of the mins, min of the maxes] -- so both shapes actually
    resolve every band. Returns (energies (n_e,), sigma)."""
    lx = torch.log(evals_x.abs().clamp_min(eps))
    ly = torch.log(evals_y.abs().clamp_min(eps))
    e_min = torch.maximum(lx[1], ly[1])
    e_max = torch.minimum(lx[-1], ly[-1])
    sigma = variance * (e_max - e_min) / n_e
    energies = torch.linspace((e_min + 2 * sigma).item(), (e_max - 2 * sigma).item(),
                              n_e, device=evals_x.device)
    return energies, sigma


def wks_coefs(evals: torch.Tensor, energies: torch.Tensor, sigma: torch.Tensor,
              eps: float = 1e-8) -> torch.Tensor:
    """Per-eigenvalue Gaussian band weights on the given energy grid: coefs[k, b] =
    exp(-(e_b - log lambda_k)^2 / 2 sigma^2). (K, n_e). The building block of both the WKS
    descriptor and the wave-kernel landmark bumps."""
    log_ev = torch.log(evals.abs().clamp_min(eps))
    return torch.exp(-(energies[None, :] - log_ev[:, None]) ** 2 / (2 * sigma ** 2 + eps))


def wks_on_grid(evals: torch.Tensor, evecs: torch.Tensor, energies: torch.Tensor,
                sigma: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """WKS descriptor on a prescribed energy grid. evals (K,), evecs (N, K) -> (N, n_e)."""
    coefs = wks_coefs(evals, energies, sigma, eps)             # (K, n_e)
    desc = (evecs ** 2) @ coefs                                # (N, n_e)
    return desc / coefs.sum(0, keepdim=True).clamp_min(eps)    # per-energy normalisation


def wks(evals: torch.Tensor, evecs: torch.Tensor, n_e: int = 100,
        variance: float = 7.0, eps: float = 1e-8) -> torch.Tensor:
    """Wave Kernel Signature per vertex (Aubry et al. 2011), auto per-shape energy range.
    evals (K,) ascending non-negative, evecs (N, K) aligned -> (N, n_e) descriptors."""
    energies, sigma = wks_grid(evals, n_e, variance, eps)
    return wks_on_grid(evals, evecs, energies, sigma, eps)
