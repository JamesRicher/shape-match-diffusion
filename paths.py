from os import path as osp

REPO_ROOT = osp.dirname(osp.abspath(__file__))
DATA_ROOT = osp.normpath(osp.join(REPO_ROOT, "..", "..", "data"))

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
