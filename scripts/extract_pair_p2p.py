"""Pull one shape pair's dense map out of several methods' maps.npz into per-method .npy.

The Blender exporter takes a predicted map as a plain .npy of target->source vertex indices
(export_correspondence_blender.py --p2p), so a comparison figure needs the same pair lifted
out of each method's maps.npz. This does that lift.

Pairs are matched BY NAME, never by position. The pair sets genuinely differ between sources:
our eval sets exclude_self (FAUST: 380 pairs) while ULRSSM enumerates the full cartesian
product (400), so index i is a different pair in each file and positional lookup silently
produces a wrong-but-plausible figure.

Without --pair it lists the pairs common to every source, ranked worst-first by our per-pair
MGE when a source carries one -- pick figure pairs by a stated rule (median, 90th percentile)
rather than by eye, and put that rule in the caption.

Example:
    # what is available, worst first
    python -m scripts.extract_pair_p2p \
        --src ours=experiments/final/faust_mpnn_512_final_cold_co/results/faust_seed0/maps.npz \
        --src ulrssm=experiments/baselines/ulrssm/faust/maps.npz

    # lift one pair for the exporter
    python -m scripts.extract_pair_p2p --pair tr_reg_080-tr_reg_085 --out-dir maps/faust \
        --src ours=... --src ulrssm=...
"""
import argparse
import os

import numpy as np


def strip_category(pair_name):
    """'crypto_Frame001-crypto_Frame002' -> 'Frame001-Frame002' (DT4D only).

    Our DT4D dataset prefixes each shape name with its category while ULRSSM keeps the bare
    frame, so the two maps.npz files share no pair names until one side is normalized. Safe
    for DT4D specifically: neither its categories nor its frame names contain an underscore,
    and applying this to an already-bare name is a no-op. Do NOT use it on SMAL, whose names
    ('cougar_01') would lose their species.
    """
    return "-".join(s.split("_", 1)[-1] for s in pair_name.split("-"))


def load_maps(path, normalize=None):
    """Read a maps.npz as (names list, {name: dense p2p}, {name: mge} or None).

    Both this repo's writer (models/mpnn_diffusion_model.py) and the patched ULRSSM one use
    the same layout: 'names' plus one ragged 'dense_%05d' per pair, addressed by POSITION in
    names. 'mge' is ours only -- ULRSSM is dense-only and stores no per-pair error.
    """
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    if normalize:
        names = [normalize(n) for n in names]
    dense = {n: z[f"dense_{i:05d}"] for i, n in enumerate(names)}
    if len(dense) != len(names):
        raise ValueError(f"{path}: duplicate pair names ({len(names)} names, {len(dense)} unique)")
    mge = dict(zip(names, z["mge"])) if "mge" in z.files else None
    return names, dense, mge


def parse_src(spec):
    """Split a --src 'label=path' argument, keeping '=' inside the path intact."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--src wants label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError(f"--src has an empty label: {spec!r}")
    return label, path


def parse_args():
    p = argparse.ArgumentParser(
        description="Lift one pair's dense map from each method's maps.npz into .npy files.")
    p.add_argument("--src", action="append", required=True, metavar="LABEL=PATH",
                   type=parse_src, help="a method's maps.npz, labelled; repeatable. The label "
                                        "names the output file and should match the --tag you "
                                        "pass to the exporter")
    p.add_argument("--pair", help="pair name '<shape_x>-<shape_y>'; omit to list common pairs")
    p.add_argument("--out-dir", help="write <label>.npy here (required with --pair)")
    p.add_argument("--top", type=int, default=20, help="pairs to list when --pair is omitted")
    p.add_argument("--strip-category", action="store_true",
                   help="DT4D: drop the '<category>_' prefix our names carry so they match "
                        "ULRSSM's bare frame names. Never use on SMAL ('cougar_01')")
    return p.parse_args()


def main():
    args = parse_args()
    norm = strip_category if args.strip_category else None
    sources = {label: load_maps(path, norm) for label, path in args.src}
    for label, path in args.src:
        names, _, mge = sources[label]
        print(f"{label:12s} {len(names):5d} pairs  mge={'yes' if mge else 'no ':3s}  {path}")

    common = set.intersection(*(set(n for n in s[0]) for s in sources.values()))
    if not common:
        raise SystemExit("no pair names common to all sources -- are these the same dataset?")
    print(f"\n{len(common)} pairs common to all {len(sources)} sources")

    if not args.pair:
        # rank by our MGE where we have one, so the listing is worst-first and a percentile
        # rule can be read straight off it; otherwise fall back to name order.
        ranked = next((sorted(common, key=lambda n: -s[2][n])
                       for s in sources.values() if s[2]), sorted(common))
        scored = next((s[2] for s in sources.values() if s[2]), None)
        print(f"\nworst first ({args.top} of {len(ranked)}):")
        for i, n in enumerate(ranked[:args.top]):
            print(f"  [{i:4d}] {n}" + (f"   mge={scored[n]:.4f}" if scored else ""))
        if scored:
            for q in (50, 75, 90):
                n = ranked[::-1][int(len(ranked) * q / 100) - 1]
                print(f"  p{q}: {n}   mge={scored[n]:.4f}")
        return

    if args.pair not in common:
        missing = [l for l, s in sources.items() if args.pair not in s[1]]
        raise SystemExit(f"pair {args.pair!r} missing from: {', '.join(missing)}")
    if not args.out_dir:
        raise SystemExit("--out-dir is required with --pair")

    os.makedirs(args.out_dir, exist_ok=True)
    sizes = {}
    for label, (_, dense, mge) in sources.items():
        p2p = dense[args.pair].astype(np.int64)
        sizes[label] = len(p2p)
        out = os.path.join(args.out_dir, f"{label}.npy")
        np.save(out, p2p)
        note = f"   mge={mge[args.pair]:.4f}" if mge else ""
        print(f"wrote {out}  (V_y={len(p2p)}, X indices {p2p.min()}..{p2p.max()}){note}")

    # every source maps the SAME target mesh, so a length disagreement means one of them is
    # from a different dataset/remesh -- the exporter would only catch it against the .off.
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"target vertex counts disagree across sources: {sizes}")


if __name__ == "__main__":
    main()
