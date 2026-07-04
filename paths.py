import os
from os import path as osp

REPO_ROOT = osp.dirname(osp.abspath(__file__))


def _resolve_data_root():
    """Locate the `data/` directory holding the datasets.

    Priority:
      1. the SHAPEMATCH_DATA_ROOT env var, if set (explicit override);
      2. `<repo>/../../data` (local layout: data two levels above the repo);
      3. `<repo>/../data`    (remote layout: data one level above the repo).
    Falls back to the local default if none exist yet.
    """
    env = os.environ.get("SHAPEMATCH_DATA_ROOT")
    if env:
        return osp.abspath(env)
    candidates = [
        osp.normpath(osp.join(REPO_ROOT, "..", "..", "data")),
        osp.normpath(osp.join(REPO_ROOT, "..", "data")),
    ]
    for c in candidates:
        if osp.isdir(c):
            return c
    return candidates[0]


DATA_ROOT = _resolve_data_root()

FAUST_DIR = osp.join(DATA_ROOT, "FAUST_r")
SMAL_DIR  = osp.join(DATA_ROOT, "SMAL_r")
SCAPE_DIR = osp.join(DATA_ROOT, "SCAPE_r")

DEFAULT_DATA_ROOTS = {
    "Faust_r": FAUST_DIR,
    "Smal_r":  SMAL_DIR,
    "Scape_r": SCAPE_DIR,
}

EXPERIMENTS_ROOT      = osp.join(REPO_ROOT, "experiments")
FROZEN_BASELINES_ROOT = osp.join(EXPERIMENTS_ROOT, "frozen_feature_baselines")
