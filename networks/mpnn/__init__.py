"""Geodesic-MPNN denoiser package (parallel to the transformer MatrixDenoiser).

Modules: geometry (graphs/RBF), intra_mpnn (geodesic GNN layer), cross_stage
(attn/mpnn A/B), state_track (in-trunk assignment state), denoiser (assembly,
registered as MPNNMatrixDenoiser). belief_prop is a planned future module; its
integration seam already exists (denoiser.bp, StateWrite's reserved channel).
"""
from networks.mpnn.denoiser import MPNNMatrixDenoiser

__all__ = ["MPNNMatrixDenoiser"]
