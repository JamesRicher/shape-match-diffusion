"""Non-visual checks for SparsePairFaustDataset (Step 2 "done when").

Verifies the item schema and that the sparse GT is a valid bijective permutation for
a sample of pairs, in both train (random FPS start) and val (fixed start) phases.

Run: python -m datasets.sparse_dataset_tests
"""
import torch

from datasets.shape_datasets import SparsePairFaustDataset
from paths import FAUST_DIR

N_SPARSE = 128
N_PAIRS = 25          # sampled pairs to check per phase


def _check_item(item, n):
    x, y = item['first'], item['second']
    assert 'gt_perm' in item and 'fps_idx' in item, "missing pair-level keys"

    for shape in (x, y):
        assert 'sparse' in shape, "missing per-shape sparse dict"
        s = shape['sparse']
        for k in ('idx', 'feat', 'verts', 'dist'):
            assert k in s, f"missing sparse key {k}"
        d_f = shape['feat'].shape[-1]
        assert s['feat'].shape == (n, d_f)
        assert s['dist'].shape == (n, n)
        assert s['verts'].shape == (n, 3)
        assert torch.unique(s['idx']).numel() == n, "idx not injective (GT not bijective)"
        # sparse dist submatrix is symmetric with a zero diagonal
        D = s['dist']
        assert torch.allclose(D, D.T, atol=1e-4) and torch.all(torch.diag(D) == 0)

    # gt_perm is a valid permutation matrix (one 1 per row/col)
    P = item['gt_perm']
    assert P.shape == (n, n)
    assert torch.all(P.sum(0) == 1) and torch.all(P.sum(1) == 1), "gt_perm not a permutation"

    # the GT actually matches: pushing the shared template positions through each
    # shape's corr must recover its sparse idx.
    K = item['fps_idx']
    assert torch.equal(x['corr'][K], x['sparse']['idx']), "first idx inconsistent with template"
    assert torch.equal(y['corr'][K], y['sparse']['idx']), "second idx inconsistent with template"


def main():
    results = []

    def check(name, fn):
        try:
            fn()
            ok, detail = True, ""
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(ok)
        print(f'[{"PASS" if ok else "FAIL"}] {name:34s}' + (f'  {detail}' if detail else ""))

    for phase in ("train", "val"):
        ds = SparsePairFaustDataset(FAUST_DIR, phase="train" if phase == "train" else "test",
                                    n_sparse=N_SPARSE)
        stride = max(1, len(ds) // N_PAIRS)

        def _run(ds=ds, stride=stride):
            for idx in range(0, len(ds), stride):
                _check_item(ds[idx], N_SPARSE)

        check(f"{phase}: schema + bijective GT ({len(ds)} pairs)", _run)

    # val is deterministic: same pair -> same FPS selection across calls
    def _val_determinism():
        ds = SparsePairFaustDataset(FAUST_DIR, phase="test", n_sparse=N_SPARSE)
        assert torch.equal(ds[7]['fps_idx'], ds[7]['fps_idx'])
    check("val FPS is deterministic", _val_determinism)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
