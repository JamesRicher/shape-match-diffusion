import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paths import FROZEN_BASELINES_ROOT

OUTPUT_ROOT = Path(FROZEN_BASELINES_ROOT)

def main():
    print(OUTPUT_ROOT)

    runs = sorted(p for p in OUTPUT_ROOT.iterdir() if (p / "pck_data.npz").exists())
    if not runs:
        print(f"no pck_data.npz files found under {OUTPUT_ROOT}")
        return

    plt.figure()
    for run_dir in runs:
        data = np.load(run_dir / "pck_data.npz")
        thresholds, pck = data["thresholds"], data["pck"]

        label = run_dir.name
        stats_path = run_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            label = f"{run_dir.name} (AUC={stats['auc']:.3f})"

        plt.plot(thresholds, pck, label=label)

    plt.xlabel("geodesic error / sqrt(area)")
    plt.ylabel("PCK")
    plt.title("Frozen feature NN — PCK curves")
    plt.xlim(0, float(thresholds[-1]))
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = OUTPUT_ROOT / "pck_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
