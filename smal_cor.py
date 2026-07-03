import os
import numpy as np
import torch
import polyscope as ps
from utils.data_utils import load_off, load_corres
from paths import SMAL_DIR

OFF_DIR = os.path.join(SMAL_DIR, "off")
K = 10

shape_a = "horse_01"
shape_b = "horse_02"

verts_a, faces_a = load_off(shape_a, SMAL_DIR)
verts_b, faces_b = load_off(shape_b, SMAL_DIR)

# corres_a is a np array of size 3889. The ith element is the image of template vertex i on 
corres_a = load_corres(shape_a, SMAL_DIR)
corres_b = load_corres(shape_b, SMAL_DIR)

extent = float(verts_a[:, 0].max() - verts_a[:, 0].min())
offset = np.array([extent * 1.3, 0.0, 0.0], dtype=verts_b.dtype)
verts_b_offset = verts_b + offset
# choose some random points on shape_a
vert_count_a = len(verts_a)
corres = np.random.choice(vert_count_a, 10, replace=False)
corres_verts_a = verts_a[corres]

ps.init()
ps.set_up_dir("neg_y_up")

mesh_a = ps.register_surface_mesh(shape_a, verts_a, faces_a, smooth_shade=False)
mesh_b = ps.register_surface_mesh(shape_b, verts_b_offset, faces_b, smooth_shade=False)
points_a = ps.register_point_cloud(shape_a, corres_verts_a)

ps.show()