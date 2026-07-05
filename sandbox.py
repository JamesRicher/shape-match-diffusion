from utils.data_utils import surface_area
from datasets import build_dataset

opt = {"name": "Faust_r", "type": "SingleFaustDataset", "phase": "test", "ret_evecs": True}
ds = build_dataset(opt)
item = ds.__getitem__(0)
print(item.get("name"))

mass = item.get('mass')
print(mass.shape)
print(mass.sum())