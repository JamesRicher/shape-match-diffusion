import re

import yaml


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
