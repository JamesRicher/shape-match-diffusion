# ULRSSM baseline maps

Predicted point-to-point maps from [ULRSSM](https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching)
(Cao et al. 2023), for the qualitative comparison figures and the baseline table column.
Generated 2026-09-02 from `Code/ULRSSM` at `/data2/home/jrr25/Documents/ULRSSM`, pulled here
with rsync. Only `maps.npz` was kept; logs, PCK plots and archived re-runs were left behind.

## File format

One `<run>/maps.npz` per run, matching this repo's own `maps.npz` layout minus the members
ULRSSM has no analogue for (no per-pair `mge`, no `sparse` / `idx_x` / `idx_y` -- it is
dense-only):

    names   (N,)      '<shape_x>-<shape_y>', in dataloader order
    dense_i (V_y,)    int32 whole-shape map Y->X, one key per pair

`dense_%05d` is addressed by POSITION in `names`. Pair sets differ between ULRSSM and our
eval (see "Self-pairs"), so **always match pairs by name, never by index**.

Written by a local patch to `ULRSSM/models/base_model.py` (`_save_maps`, gated on
`val: save_maps: true`). That patch is not upstream; `results/` is gitignored there, which is
why these live here instead.

## Runs

| run | checkpoint | refine | pairs | self-pairs | status |
| --- | --- | --- | --- | --- | --- |
| `faust`            | faust.pth       | none | 400  | 20 | **verified** 0.0159 vs published 1.6 |
| `scape`            | scape.pth       | none | 400  | 20 | **verified** 0.0192 vs published 1.9 |
| `dt4d_intra_class` | dt4d.pth        | none | 825  | 85 | **verified** 0.0091 vs published 0.9 |
| `smal`             | smal.pth        | 15   | 400  | 20 | **verified** reproduces published |
| `dt4d_inter_class` | dt4d.pth        | 15   | 1200 |  0 | **verified** reproduces published |
| `shrec19`          | faust_scape.pth | 12   | 407  |  0 | unchecked against published |
| `faust_on_scape`         | faust.pth       | none | 400 | 20 | **DO NOT USE** |
| `scape_on_faust`         | scape.pth       | none | 400 | 20 | **DO NOT USE** |
| `faust_scape_on_faust`   | faust_scape.pth | none | 400 | 20 | **DO NOT USE** |
| `faust_scape_on_scape`   | faust_scape.pth | none | 400 | 20 | **DO NOT USE** |
| `faust_on_shrec19`       | faust.pth       | 12   | 407 |  0 | suspect, see below |
| `scape_on_shrec19`       | scape.pth       | 12   | 407 |  0 | suspect: 0.0782 vs published 6.7 |

The six `*_on_*` runs are cross-dataset configs written for this project
(`ULRSSM/options/test/cross/`), not shipped by ULRSSM. Every *shipped* config reproduces the
paper; both classes of failure below are in the custom ones.

### Why the four "DO NOT USE" runs are wrong

They were generated from the in-distribution templates (`faust.yaml`, `scape.yaml`), which
carry no test-time refinement -- but ULRSSM's cross-dataset numbers use it. Sorting the
shipped configs makes the rule obvious: in-distribution (faust, scape, dt4d_intra) has no
`refine`, while every distribution-shift setting (faust_a, shrec19, smal, dt4d_inter, topkids,
shrec20, shrec16_*) sets `refine: 12`-`25` plus the `fmap_net` and `train` blocks it needs.
`scape_on_faust` came out 0.0458 against a published 1.6 as a result.

To fix, rebuild them from `options/test/faust_a.yaml` -- same benchmark family, generalisation
setting, and it already carries `refine: 12`, `fmap_net: RegularizedFMNet` and the Adam +
SURFMNet/SquaredFrobenius `train` block that `refine()` calls into. Adding `refine:` alone to a
faust/scape-derived config will raise a KeyError on `networks['fmap_net']`.

The two `*_on_shrec19` runs *do* have the full apparatus (inherited from `shrec19.yaml`), but
`scape_on_shrec19` still lands 17% short of the published 6.7. Ruled out: geodesic
normalisation (SHREC19_r diameters ~1.63, in line with FAUST 1.75 / SCAPE 1.66) and the pair
set (430 `.map` files minus exactly the 23 involving shape 40 = 407). Most likely the
`refine: 12` budget is tuned for the `faust_scape.pth` starting point it ships with.

## Protocol differences vs our eval

Two independent gaps; both need stating in the experimental setup.

**Self-pairs.** ULRSSM enumerates the full cartesian product *including* the diagonal, and
pools per-vertex errors across all pairs. Our configs set `exclude_self: true`. On FAUST the
20 identity pairs score 1.1e-5 and dilute the reported mean by 5.3% (0.0159 pooled vs 0.0167
on non-diagonal pairs alone). Affects FAUST, SCAPE, SMAL, DT4D-intra. Does **not** affect
DT4D-inter (no diagonal by construction) or SHREC19 (fixed pairwise `.map` list).

Since baseline table entries are copied from published papers, the fixed protocol is theirs --
so matching means evaluating *ours* with self-pairs included, not re-scoring theirs.

**Test-time adaptation.** ULRSSM refines its weights on each test pair (12-15 unsupervised
steps, weights restored between pairs) on SMAL, DT4D-inter and SHREC19; none on FAUST, SCAPE,
DT4D-intra. We do no gradient adaptation anywhere -- our inference-time compute is the
densifier, the BP post-process and best-of-K sampling. Note this lands on the benchmarks where
the BP post-process result is strongest.

## Reproducing

    cd Code/ULRSSM && conda activate fmnet          # NOT shapematch; ULRSSM needs py3.8 + trimesh
    python test.py --opt options/test/<dataset>.yaml

`maps.npz` is written to `results/<name>/` where `<name>` is the config's `name:` field, only
when `val: save_maps: true`. Re-running archives the previous results dir rather than
overwriting it. Maps are written after the full loop, so an interrupted run produces nothing.
