"""DiffusionNet feature extractor.

Ported from ULRSSM's networks/diffusion_network.py (itself adapted from the official
nmwsharp/diffusion-net), specialised to this codebase:

  * SINGLE-SHAPE / UNBATCHED. The diffusion model runs the extractor one shape at a time
    (`_sparse_inputs`), so every op here works on 2-D tensors ([V, C]) rather than ULRSSM's
    batched [B, V, C]. This drops the per-shape gradX/gradY batch-indexing gymnastics.
  * OPERATORS ARE PRE-CACHED. Our dataset already emits evecs / evals / mass / gradX / gradY
    per item (ret_evecs=True, dataset_bases._load_ops), so `forward` TAKES the operators
    instead of recomputing them via get_all_operators. Enable ret_evecs on train/val/test.
  * spectral diffusion only (the default); implicit_dense is not ported.

`DiffusionNetExtractor` is the registered network. It sets `needs_operators = True` so the
model calls `extract(shape_dict, idx)` (operators live in the shape dict) instead of the GCN
extractor's `extract(verts, dist, idx)`. Non-learned operators, so no separate cache is built.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import NETWORK_REGISTRY


# --------------------------------------------------------------------------- #
# input descriptors
# --------------------------------------------------------------------------- #
def compute_hks_autoscale(evals, evecs, count=16):
    """Heat-kernel signature with auto-scaled times. evals [K], evecs [V, K] -> [V, count]."""
    scales = torch.logspace(-2.0, 0.0, steps=count, device=evals.device, dtype=evals.dtype)
    power = torch.exp(-evals.unsqueeze(0) * scales.unsqueeze(-1))       # [S, K]
    return torch.einsum('vk,sk->vs', evecs * evecs, power)             # [V, S]


# --------------------------------------------------------------------------- #
# building blocks (unbatched)
# --------------------------------------------------------------------------- #
class LearnedTimeDiffusion(nn.Module):
    """Per-channel learned-time heat diffusion in the LBO spectral domain."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.diffusion_time = nn.Parameter(torch.zeros(in_channels))

    def forward(self, feat, mass, evals, evecs):
        # feat [V, C]; mass [V]; evals [K]; evecs [V, K]
        with torch.no_grad():
            self.diffusion_time.clamp_(min=1e-8)                        # positive half-space
        feat_spec = evecs.transpose(-2, -1) @ (feat * mass.unsqueeze(-1))  # [K, C]
        decay = torch.exp(-evals.unsqueeze(-1) * self.diffusion_time.unsqueeze(0))  # [K, C]
        return evecs @ (decay * feat_spec)                             # [V, C]


class SpatialGradientFeatures(nn.Module):
    """Learned complex-linear dot products between spatial gradient components."""
    def __init__(self, in_channels, with_gradient_rotations=True):
        super().__init__()
        self.with_gradient_rotations = with_gradient_rotations
        if with_gradient_rotations:
            self.A_re = nn.Linear(in_channels, in_channels, bias=False)
            self.A_im = nn.Linear(in_channels, in_channels, bias=False)
        else:
            self.A = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, feat_in):
        # feat_in [V, C, 2]  (real, imag gradient parts)
        if self.with_gradient_rotations:
            real_b = self.A_re(feat_in[..., 0]) - self.A_im(feat_in[..., 1])
            imag_b = self.A_re(feat_in[..., 0]) + self.A_im(feat_in[..., 1])
        else:
            real_b = self.A(feat_in[..., 0])
            imag_b = self.A(feat_in[..., 1])
        out = feat_in[..., 0] * real_b + feat_in[..., 1] * imag_b
        return torch.tanh(out)                                        # [V, C]


class MiniMLP(nn.Sequential):
    """Small per-vertex MLP (Linear over the channel dim)."""
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU):
        super().__init__()
        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))
            if dropout and i > 0:
                self.add_module(f'dropout_{i:03d}', nn.Dropout(p=0.5))
            self.add_module(f'linear_{i:03d}', nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if not is_last:
                self.add_module(f'act_{i:03d}', activation())


class DiffusionNetBlock(nn.Module):
    """One DiffusionNet block: diffuse -> spatial-gradient features -> MLP, with a skip."""
    def __init__(self, channels, mlp_hidden_channels, dropout=True,
                 with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()
        self.channels = channels
        self.with_gradient_features = with_gradient_features
        self.diffusion = LearnedTimeDiffusion(channels)
        mlp_in = 2 * channels
        if with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                channels, with_gradient_rotations=with_gradient_rotations)
            mlp_in += channels
        self.mlp = MiniMLP([mlp_in] + list(mlp_hidden_channels) + [channels], dropout=dropout)

    def forward(self, feat_in, mass, evals, evecs, gradX, gradY):
        # feat_in [V, C]; gradX/gradY sparse [V, V]
        feat_diffuse = self.diffusion(feat_in, mass, evals, evecs)
        if self.with_gradient_features:
            grad_x = torch.sparse.mm(gradX, feat_diffuse)             # [V, C]
            grad_y = torch.sparse.mm(gradY, feat_diffuse)
            feat_grad = torch.stack((grad_x, grad_y), dim=-1)         # [V, C, 2]
            feat_grad = self.gradient_features(feat_grad)
            feat_combined = torch.cat((feat_in, feat_diffuse, feat_grad), dim=-1)
        else:
            feat_combined = torch.cat((feat_in, feat_diffuse), dim=-1)
        return self.mlp(feat_combined) + feat_in                     # skip connection


class DiffusionNetCore(nn.Module):
    """First linear -> N diffusion blocks -> last linear. Per-vertex features."""
    def __init__(self, in_channels, out_channels, hidden_channels=128, n_block=4,
                 mlp_hidden_channels=None, dropout=False,
                 with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()
        if not mlp_hidden_channels:
            mlp_hidden_channels = [hidden_channels, hidden_channels]
        self.first_linear = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            DiffusionNetBlock(hidden_channels, mlp_hidden_channels, dropout=dropout,
                              with_gradient_features=with_gradient_features,
                              with_gradient_rotations=with_gradient_rotations)
            for _ in range(n_block))
        self.last_linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, mass, evals, evecs, gradX, gradY):
        x = self.first_linear(x)
        for block in self.blocks:
            x = block(x, mass, evals, evecs, gradX, gradY)
        return self.last_linear(x)                                   # [V, out_channels]


# --------------------------------------------------------------------------- #
# extractor (registered)
# --------------------------------------------------------------------------- #
def _random_rotation(deg_xyz, device, dtype):
    """Random rotation matrix, each axis drawn uniformly from [-deg, +deg] (degrees).

    Follows ULRSSM's get_random_rotation / euler_angles_to_rotation_matrix: per-axis Euler
    angles composed R_z @ R_y @ R_x. Its default (0, 90, 0) is yaw only, which is the axis
    SMAL_r actually varies along.
    """
    t = [(torch.rand((), device=device, dtype=dtype) * 2 - 1) * (float(d) * math.pi / 180.0)
         for d in deg_xyz]
    one, zero = torch.ones((), device=device, dtype=dtype), torch.zeros((), device=device, dtype=dtype)
    cx, sx, cy, sy, cz, sz = (torch.cos(t[0]), torch.sin(t[0]), torch.cos(t[1]),
                              torch.sin(t[1]), torch.cos(t[2]), torch.sin(t[2]))
    stack = lambda rows: torch.stack([torch.stack(r) for r in rows])
    R_x = stack([[one, zero, zero], [zero, cx, -sx], [zero, sx, cx]])
    R_y = stack([[cy, zero, sy], [zero, one, zero], [-sy, zero, cy]])
    R_z = stack([[cz, -sz, zero], [sz, cz, zero], [zero, zero, one]])
    return R_z @ R_y @ R_x


@NETWORK_REGISTRY.register()
class DiffusionNetExtractor(nn.Module):
    """Per-vertex DiffusionNet descriptor, sampled at the FPS points.

    Interface mirrors GCNFeatureExtractor but consumes the shape dict (for its cached spectral
    operators) rather than (verts, dist): the model branches on ``needs_operators``.

    Args:
        out_dim: per-point feature dim (= denoiser feat_dim).
        input_type: 'xyz' (extrinsic, 3-dim; best within-dataset symmetry) or 'hks'
            (intrinsic, hks_count-dim; transfers across datasets, symmetry-ambiguous).
        hidden: DiffusionNet width. n_block: number of diffusion blocks.
        k_eig: LBO eigenpairs used for spectral diffusion (<= dataset num_evecs).
        hks_count: number of HKS bands when input_type == 'hks'.
        rot_aug: per-axis (x, y, z) degrees of random rotation applied to xyz input during
            TRAINING only, each drawn uniformly from [-deg, +deg]; (0, 0, 0) disables it.
            Use [0, 90, 0] to match ULRSSM, whose SMAL model relies on it: xyz is extrinsic,
            so without augmentation the network keys on a per-shape frame that is arbitrary
            (SMAL_r shapes differ by ~108 degrees of median yaw). Ignored for hks, which is
            intrinsic and already rotation-invariant.

            Rotation ONLY -- deliberately not ULRSSM's accompanying noise and rescaling. The
            spectral operators (mass, evals, evecs, gradX, gradY) are cached per mesh and are
            rotation-invariant, so rotating the input stays exactly consistent with them;
            jittering or rescaling the vertices would not.
        dropout / with_gradient_features / with_gradient_rotations: DiffusionNet options.
    """
    needs_operators = True   # tells the model to call extract(shape_dict, idx)

    def __init__(self, out_dim=128, input_type='xyz', hidden=128, n_block=4, k_eig=128,
                 hks_count=16, rot_aug=(0.0, 0.0, 0.0), dropout=False,
                 with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()
        if input_type not in ('xyz', 'hks'):
            raise ValueError(f"input_type must be 'xyz' or 'hks', got {input_type!r}")
        self.input_type = input_type
        rot_aug = (rot_aug,) * 3 if isinstance(rot_aug, (int, float)) else tuple(rot_aug)
        if len(rot_aug) != 3:
            raise ValueError(f'rot_aug must be a scalar or 3 per-axis degrees, got {rot_aug!r}')
        if any(rot_aug) and input_type != 'xyz':
            raise ValueError(f"rot_aug is only meaningful for input_type 'xyz', got {input_type!r}")
        self.rot_aug = rot_aug
        self.k_eig = k_eig
        self.hks_count = hks_count
        in_channels = 3 if input_type == 'xyz' else hks_count
        self.net = DiffusionNetCore(
            in_channels, out_dim, hidden_channels=hidden, n_block=n_block, dropout=dropout,
            with_gradient_features=with_gradient_features,
            with_gradient_rotations=with_gradient_rotations)

    def _device(self):
        return self.net.first_linear.weight.device

    def _per_vertex(self, shape):
        """Run DiffusionNet over the whole mesh -> (V, out_dim), using the shape's cached ops."""
        dev = self._device()
        evecs = shape['evecs'].to(dev).float()                       # [V, K]
        evals = shape['evals'].to(dev).float()                       # [K]
        mass = shape['mass'].to(dev).float()                         # [V]
        gradX = shape['gradX'].to(dev)                               # sparse [V, V]
        gradY = shape['gradY'].to(dev)
        k = min(self.k_eig, evecs.shape[1])
        evecs, evals = evecs[:, :k], evals[:k]

        if self.input_type == 'xyz':
            x = shape['verts'].to(dev).float()                       # [V, 3]
            if self.training and any(self.rot_aug):                  # independent draw per shape
                x = x @ _random_rotation(self.rot_aug, dev, x.dtype).T
        else:
            x = compute_hks_autoscale(evals, evecs, self.hks_count)  # [V, hks_count]
        return self.net(x, mass, evals, evecs, gradX, gradY)         # [V, out_dim]

    def extract(self, shape, idx):
        """(1, n, out_dim) descriptors at the FPS points `idx` (full-mesh vertex indices)."""
        feats = self._per_vertex(shape)
        return feats[idx.to(self._device()).long()].unsqueeze(0)

    def extract_dense(self, shape):
        """(V, out_dim) per-vertex descriptors for a densifier data term."""
        return self._per_vertex(shape)
