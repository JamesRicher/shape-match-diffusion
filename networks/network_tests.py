import torch

from networks import build_network


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, in_dim, dim, heads = 2, 16, 8, 32, 4
    enc = build_network({
        "type": "ShapeMatchingEncoder",
        "in_dim": in_dim,
        "dim": dim,
        "heads": heads,
        "depth": 2,
    }).eval()

    fx = torch.randn(B, N, in_dim)
    fy = torch.randn(B, N, in_dim)

    ex, ey = enc(fx, fy)
    # swap inputs; with inter_bias=None there is nothing to transpose
    ex_s, ey_s = enc(fy, fx)

    err_x = (ex - ey_s).abs().max().item()
    err_y = (ey - ex_s).abs().max().item()
    print(f"output shapes: {tuple(ex.shape)}, {tuple(ey.shape)}")
    print(f"symmetry residuals (want ~0): {err_x:.2e}, {err_y:.2e}")

    # symmetry with a non-trivial inter bias: swap inputs AND transpose the bias
    bias = torch.randn(B, N, N)
    ax, ay = enc(fx, fy, inter_bias=bias)
    bx, by = enc(fy, fx, inter_bias=bias.transpose(-1, -2))
    print("symmetry residuals w/ bias: "
          f"{(ax - by).abs().max().item():.2e}, {(ay - bx).abs().max().item():.2e}")
