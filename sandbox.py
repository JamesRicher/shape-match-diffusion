from utils.data_utils import surface_area
from utils.options import load_yaml
from datasets import build_dataset
from models import build_model
from torch.utils.data import DataLoader

opt = {"name": "Faust_r", "type": "PairFaustDataset", "phase": "test", "ret_evecs": True}
ds = build_dataset(opt)
item = ds.__getitem__(0)['first']
print(item.get("name"))

mass = item.get('mass')
print(mass.shape)
print(mass.sum())

# lightweight model build from the training config: fill the one runtime field
# (encoder.in_dim is null in the yaml) and construct in inference mode.
model_opt = load_yaml("configs/faust_shape_matching.yaml")
model_opt['is_train'] = False
model_opt['networks']['encoder']['in_dim'] = int(item['feat'].shape[-1])
model = build_model(model_opt)

def _single_collate(batch):
    """batch_size=1 collate that returns the sample untouched.

    The shape pairs hold variable-size and sparse tensors (operators), which the
    default collate cannot stack, so we train one pair at a time.
    """
    return batch[0]

val_loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_single_collate, num_workers=0)
model.validation(val_loader)