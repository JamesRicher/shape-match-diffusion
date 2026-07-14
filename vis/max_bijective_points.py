"""Report the maximum bijective sparse-FPS point count for a dataset (headless).

The .vts template->vertex map is many-to-one per shape, so only so many FPS points can
be distinct on BOTH shapes of a pair at once (see utils.data_utils.bijective_fps_order).
This scans pairs and prints the per-pair maxima plus the MIN across pairs -- the largest
n_sparse for which a bijective sparse GT exists for every scanned pair. No polyscope, so
it runs on a headless cluster (unlike vis.sparse_fps_pair_vis).

Examples:
    python -m vis.max_bijective_points --type SingleFaustDataset --name Faust_r --phase train
    python -m vis.max_bijective_points --type SingleSmalDataset  --name Smal_r  --phase train --category true
    python -m vis.max_bijective_points --type SingleScapeDataset --name Scape_r --phase train
"""
import argparse
import itertools
import sys

import numpy as np

from datasets import build_dataset
from utils.data_utils import bijective_fps_order


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", default="SingleFaustDataset",
                        help="Dataset class name registered in DATASET_REGISTRY")
    parser.add_argument("--name", default="Faust_r", help="Dataset name key for the data root")
    parser.add_argument("--phase", default="train", help="train/test/full (default: train)")
    parser.add_argument("--category", default=None,
                        help="SMAL only: true/false (train_cat.txt vs train.txt)")
    parser.add_argument("--start", type=int, default=0, help="FPS start (template index)")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="cap the number of pairs scanned (0 = all ordered pairs)")
    args = parser.parse_args(argv)

    opt = {"type": args.type, "name": args.name, "phase": args.phase,
           "ret_faces": False, "ret_feats": False, "ret_corr": True,
           "ret_dist": False, "ret_evecs": False}
    if args.category is not None:
        opt["category"] = args.category.lower() in ("1", "true", "yes")

    ds = build_dataset(opt)
    n = len(ds)
    if n < 2:
        raise RuntimeError(f"Dataset has only {n} shapes; need at least 2.")

    # cache each shape's verts + template->vertex map once
    shapes = [{"name": ds[i].get("name", i),
               "verts": _to_numpy(ds[i]["verts"]).astype(np.float32),
               "corr": _to_numpy(ds[i]["corr"]).astype(np.int64)} for i in range(n)]

    pairs = [(i, j) for i, j in itertools.product(range(n), repeat=2) if i != j]
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    overall_min, argmin = None, None
    print(f"{args.name} [{args.type}] phase={args.phase} "
          f"category={opt.get('category', 'n/a')}: {n} shapes, {len(pairs)} pairs")
    for i, j in pairs:
        a, b = shapes[i], shapes[j]
        m = len(bijective_fps_order(a["verts"], a["corr"], b["corr"], args.start))
        if overall_min is None or m < overall_min:
            overall_min, argmin = m, (a["name"], b["name"])

    print(f"template size T = {shapes[0]['corr'].shape[0]}")
    print(f"MIN bijective points across pairs (start={args.start}): {overall_min}"
          f"   (worst pair: {argmin[0]} -> {argmin[1]})")
    print(f"=> largest safe n_sparse for every scanned pair: {overall_min}")


if __name__ == "__main__":
    main(sys.argv[1:])
