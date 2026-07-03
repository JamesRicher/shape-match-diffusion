from datasets.shape_datasets import *
import polyscope as ps
import numpy as np

from paths import FAUST_DIR as faust_root, SMAL_DIR as smal_root, SCAPE_DIR as scape_root

k = 10

ps.init()

faust_dataset = SingleFaustDataset(data_root=faust_root, phase="full")
item = faust_dataset.__getitem__(90)

smal_dataset = SingleSmalDataset(data_root=smal_root, phase="train", category=False)
scape_dataset = SingleScapeDataset(data_root=scape_root, phase="train", ret_dist=False, ret_feats=False)
item = scape_dataset.__getitem__(10)

faust_pair_ds = PairFaustDataset(faust_root, phase="full")
item = faust_pair_ds.__getitem__(876)['first']

smal_pair_ds = PairSmalDataset(smal_root, phase="train")
item = smal_pair_ds.__getitem__(19)['first']

scape_pair_ds = PairScapeDataset(scape_root, "train")
item = scape_pair_ds.__getitem__(19)

shape_a = item['first']
shape_b = item['second']

verts_a = shape_a['verts']
faces_a = shape_a['faces']
verts_b = shape_b['verts']
faces_b = shape_b['faces']

extent = float(verts_a[:, 0].max() - verts_a[:, 0].min())
offset = np.array([extent * 2, 0.0, 0.0])
verts_b_offset = verts_b + offset

template_len = len(shape_a["corr"])
indices = np.random.choice(template_len, k)

points_a = verts_a[shape_a["corr"][indices]]
points_b = verts_b[shape_b["corr"][indices]] + offset

ps.register_surface_mesh(shape_a["name"], verts_a, faces_a, smooth_shade=False)
ps.register_surface_mesh(shape_b["name"], verts_b_offset, faces_b, smooth_shade=False)

ps.register_point_cloud(shape_a["name"], points_a)
ps.register_point_cloud(shape_b["name"], points_b)
#ps.set_up_dir("neg_y_up")
ps.show()