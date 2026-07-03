import os
import numpy as np
import torch
import polyscope as ps
from utils.data_utils import load_off
from paths import SMAL_DIR

FEATS_DIR = os.path.join(SMAL_DIR, "feats_norm")

shape_a = "horse_01"
shape_b = "horse_02"


def load_feats_pt(name: str) -> np.ndarray:
    t = torch.load(os.path.join(FEATS_DIR, f"{name}.npy"),
                   map_location="cpu", weights_only=False)
    return t.numpy()

verts_a, faces_a = load_off(shape_a, SMAL_DIR)
verts_b, faces_b = load_off(shape_b, SMAL_DIR)
feats_a = load_feats_pt(shape_a)
feats_b = load_feats_pt(shape_b)
assert feats_a.shape[0] == verts_a.shape[0]
assert feats_b.shape[0] == verts_b.shape[0]
assert feats_a.shape[1] == feats_b.shape[1]
n_channels = feats_a.shape[1]

print(len((verts_a)))
print(len((verts_b)))

# side-by-side layout: offset shape B along X by the combined extent
extent = float(verts_a[:, 0].max() - verts_a[:, 0].min())
offset = np.array([extent * 1.3, 0.0, 0.0], dtype=verts_b.dtype)
verts_b_offset = verts_b + offset

ps.init()
ps.set_up_dir("neg_y_up")
mesh_a = ps.register_surface_mesh(shape_a, verts_a, faces_a, smooth_shade=True)
mesh_b = ps.register_surface_mesh(shape_b, verts_b_offset, faces_b, smooth_shade=True)

# add each feature channel as a scalar quantity on both meshes so the user
# can flip through them in the polyscope UI. use a shared symmetric range
# across both shapes so colors are directly comparable per channel.
for c in range(n_channels):
    name = f"feat_{c:03d}"
    vmax = float(max(np.abs(feats_a[:, c]).max(), np.abs(feats_b[:, c]).max()))
    vrange = (-vmax, vmax)
    enabled = (c == 0)
    mesh_a.add_scalar_quantity(name, feats_a[:, c], cmap="coolwarm",
                               vminmax=vrange, enabled=enabled)
    mesh_b.add_scalar_quantity(name, feats_b[:, c], cmap="coolwarm",
                               vminmax=vrange, enabled=enabled)

ps.show()
