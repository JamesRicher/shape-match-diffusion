import os.path as osp
import re

import yaml

from paths import EXPERIMENTS_ROOT


# PyYAML's default resolver treats scientific notation without a decimal point
# (e.g. ``1e-3``) as a *string*, which silently breaks learning rates. Install a
# stricter float resolver so ``1e-3`` / ``1e-4`` parse as floats.
_FLOAT_RE = re.compile(r'''^(?:
     [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN)
    )$''', re.X)


class _YamlLoader(yaml.SafeLoader):
    """SafeLoader with scientific-notation float support."""


_YamlLoader.add_implicit_resolver('tag:yaml.org,2002:float', _FLOAT_RE, list('-+0123456789.'))


def load_yaml(path):
    """Load a YAML config file into a nested dict."""
    with open(path, 'r') as f:
        return yaml.load(f, Loader=_YamlLoader)


def resolve_experiment_paths(opt, resume=None):
    """Populate ``opt['path']`` with the standard experiment layout under
    ``experiments/<name>/``, without clobbering anything already set.

    Layout:
        experiments/<name>/   experiment_info.json (config + network stats)
        models/               checkpoints (``latest.pth`` = resumable, ``final.pth`` = final-epoch)
        results/              all run artifacts: pck.png/pck.npy, test stats.json

    Shared by ``train.py`` and ``evaluate.py`` so both agree on where things live.
    ``resume`` (if given) sets ``path['resume_state']`` (checkpoint to load).
    """
    exp_dir = osp.join(EXPERIMENTS_ROOT, opt['name'])
    path = opt.setdefault('path', {})
    path.setdefault('experiment_root', exp_dir)
    path.setdefault('models', osp.join(exp_dir, 'models'))
    path.setdefault('results', osp.join(exp_dir, 'results'))
    if resume is not None:
        path['resume_state'] = resume
    return opt
