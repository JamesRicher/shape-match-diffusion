"""Inspect the trainable parameters of a saved MatrixDenoiser checkpoint.

Prints the full parameter inventory (name, shape, count) and then the trained
attention-bias parameters in their effective units: the intra self-attention
geodesic gammas (softplus of raw_gamma, per layer x head) and the inter
cross-attention log-P_t gammas (raw, zero-init).

Usage:
    python debug/inspect_denoiser_params.py experiments/faust_matrix_diffusion_v2/models/final.pth
    python debug/inspect_denoiser_params.py <ckpt> --no-inventory   # bias tables only
"""
import argparse
import math

import torch
import torch.nn.functional as F


def load_denoiser_sd(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt['networks']['denoiser']


def print_inventory(sd: dict) -> None:
    total = 0
    for name, p in sd.items():
        total += p.numel()
        print(f"{name:60s} {str(tuple(p.shape)):18s} {p.numel():>10,}")
    print(f"\n{'TOTAL':60s} {'':18s} {total:>10,}")


def bias_tables(sd: dict) -> None:
    intra_layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('intra_bias.')})
    heads = sd['intra_bias.0.raw_gamma'].numel()
    header = f"{'layer':>5} | " + " | ".join(f"head {h}" for h in range(heads))

    init = torch.logspace(math.log10(0.1), math.log10(10.0), heads)
    print("\n=== intra (self-attn) GeodesicKernelBias: gamma = softplus(raw_gamma) ===")
    print("bias = -gamma_h * D^2; larger gamma = tighter geodesic locality")
    print("init:  " + " | ".join(f"{v:6.3f}" for v in init.tolist()))
    print(header)
    for l in intra_layers:
        g = F.softplus(sd[f'intra_bias.{l}.raw_gamma'])
        print(f"{l:>5} | " + " | ".join(f"{v:6.3f}" for v in g.tolist()))

    print("\n=== inter (cross-attn) LogAssignmentBias: gamma (raw, zero-init) ===")
    print("bias = gamma_h * log(P_t); 0 ignore, ~1 product-of-experts, >1 hard routing, <0 anti-prior")
    print(header)
    for l in intra_layers:
        g = sd[f'inter_bias.{l}.gamma']
        print(f"{l:>5} | " + " | ".join(f"{v:+6.3f}" for v in g.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('ckpt', help='path to a saved checkpoint (final.pth / latest.pth)')
    ap.add_argument('--no-inventory', action='store_true', help='skip the full parameter listing')
    args = ap.parse_args()

    sd = load_denoiser_sd(args.ckpt)
    if not args.no_inventory:
        print_inventory(sd)
    bias_tables(sd)


if __name__ == '__main__':
    main()
