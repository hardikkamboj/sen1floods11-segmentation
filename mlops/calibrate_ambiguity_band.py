"""Empirically calibrate the classical-method ambiguity band (Figure A).

The cascaded inference pipeline (Novelty 1) routes pixels to the deep model
only when the classical method (-13.45 dB VV threshold) is uncertain. The
question is: how do we *define* "uncertain"?

This script derives the answer from data, not from a hand-picked constant.
For every pixel in the test set, we measure two things:
  1. distance from the fixed-dB threshold (|VV - (-13.45)|)
  2. whether the classical prediction matches the ground truth label

Bucketing pixels by distance and measuring per-bucket accuracy yields a
calibration curve. The empirical ambiguity band is the distance range where
classical accuracy falls below a chosen confidence target (default 0.85).

Outputs
-------
- ``mlops/calibration.json`` — the empirical band, bucket stats, hyperparams
- ``mlops/figures/ambiguity_calibration.png`` — Figure A for the report
- A ClearML Task (project ``Sen1Floods11/Calibration``) holding both above

Usage
-----
    python mlops/calibrate_ambiguity_band.py \\
        --s1-dir    /content/sen1floods11/S1 \\
        --label-dir /content/sen1floods11/Labels \\
        --split-csv /content/sen1floods11/splits/flood_test_data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from clearml import Task

# Threshold from the progress report's classical baseline study.
# This is what the cascade trusts on confident pixels.
DB_THRESHOLD = -13.45


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_chip(s1_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vv_db, label, valid_mask) — all (H, W) arrays. NaN VV → 0 dB
    (rare; speckle nulls). Invalid pixels (label == -1) get masked."""
    with rasterio.open(s1_path) as src:
        s1 = src.read().astype(np.float32)            # (2, H, W)
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.float32)        # (H, W)

    vv = np.nan_to_num(s1[0], nan=0.0)
    valid_mask = (label != -1)
    label = np.clip(label, 0, 1).astype(np.uint8)
    return vv, label, valid_mask


def calibrate(
    s1_dir: Path,
    label_dir: Path,
    split_csv: Path,
    db_threshold: float,
    bin_edges: np.ndarray,
    target_accuracy: float,
):
    """Walk the test set, accumulate (distance, correct?) for every valid
    pixel, then bucket by distance to compute per-bin accuracy.

    We stream pixel histograms instead of materializing every-pixel arrays —
    a 90-chip test set is ~24M pixels, and we need this to fit comfortably
    in memory on Colab.
    """
    n_bins = len(bin_edges) - 1
    correct_per_bin = np.zeros(n_bins, dtype=np.int64)
    count_per_bin   = np.zeros(n_bins, dtype=np.int64)

    pairs = []
    with open(split_csv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s1_name, lbl_name = [s.strip() for s in line.split(",")]
            pairs.append((s1_name, lbl_name))

    print(f"Calibrating on {len(pairs)} chips...")
    for i, (s1_name, lbl_name) in enumerate(pairs, 1):
        vv, label, valid = load_chip(s1_dir / s1_name, label_dir / lbl_name)

        # Classical prediction on every pixel.
        classical_pred = (vv < db_threshold).astype(np.uint8)
        correct        = (classical_pred == label)

        # Distance from the threshold — the proxy for classical confidence.
        distance = np.abs(vv - db_threshold)

        # Restrict to valid pixels, then histogram by distance bin.
        d   = distance[valid]
        ok  = correct[valid]
        idx = np.digitize(d, bin_edges) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        np.add.at(count_per_bin,   idx, 1)
        np.add.at(correct_per_bin, idx, ok.astype(np.int64))

        if i % 20 == 0:
            print(f"  processed {i}/{len(pairs)}")

    accuracy_per_bin = correct_per_bin / np.maximum(count_per_bin, 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # ── Determine the empirical ambiguity band ───────────────────────────────
    # The band is the contiguous low-distance region where accuracy stays
    # below ``target_accuracy``. We walk outward from 0 dB until accuracy
    # crosses the target — that crossing point is the band edge.
    band_edge = float(bin_edges[-1])  # default: full range is uncertain
    for c, acc in zip(bin_centers, accuracy_per_bin):
        if acc >= target_accuracy:
            band_edge = float(c)
            break

    return {
        "bin_edges":        bin_edges.tolist(),
        "bin_centers":      bin_centers.tolist(),
        "count_per_bin":    count_per_bin.tolist(),
        "accuracy_per_bin": accuracy_per_bin.tolist(),
        "band_edge_db":     band_edge,
        "target_accuracy":  target_accuracy,
        "db_threshold":     db_threshold,
    }


def make_figure(stats: dict, out_path: Path):
    """Figure A — distance-to-threshold vs classical accuracy.
    The shaded region is the empirical ambiguity band.
    """
    bin_centers = np.array(stats["bin_centers"])
    accuracy    = np.array(stats["accuracy_per_bin"])
    counts      = np.array(stats["count_per_bin"])
    band_edge   = stats["band_edge_db"]
    target_acc  = stats["target_accuracy"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Accuracy curve (left axis)
    ax1.plot(bin_centers, accuracy, marker="o", color="#1f77b4",
             label="Classical accuracy")
    ax1.axhline(target_acc, color="#d62728", linestyle="--", alpha=0.7,
                label=f"Target accuracy ({target_acc:.2f})")
    ax1.axvspan(0, band_edge, color="#d62728", alpha=0.12,
                label=f"Ambiguity band (0–{band_edge:.2f} dB)")
    ax1.set_xlabel("Distance from threshold |VV − (−13.45)| (dB)")
    ax1.set_ylabel("Classical accuracy", color="#1f77b4")
    ax1.set_ylim(0.4, 1.02)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    # Pixel-count histogram (right axis) — context for how many pixels
    # actually fall in each distance bin.
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts, width=np.diff(stats["bin_edges"]),
            color="#7f7f7f", alpha=0.18, label="Pixel count")
    ax2.set_ylabel("Pixel count (test set)", color="#7f7f7f")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title(
        "Classical accuracy collapses near the dB threshold\n"
        f"Empirical ambiguity band: ±{band_edge:.2f} dB around "
        f"VV = {stats['db_threshold']} dB"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s1-dir",    required=True)
    p.add_argument("--label-dir", required=True)
    p.add_argument("--split-csv", required=True,
                   help="Path to flood_test_data.csv (calibrate on test set)")
    p.add_argument("--db-threshold",   type=float, default=DB_THRESHOLD)
    p.add_argument("--target-accuracy", type=float, default=0.85,
                   help="Confidence target — bins below this go in the ambiguity band")
    p.add_argument("--bin-width-db",   type=float, default=0.5,
                   help="Width of each distance bin in dB")
    p.add_argument("--max-distance-db", type=float, default=10.0,
                   help="Maximum |VV − threshold| to histogram (dB)")
    p.add_argument("--out-json",   default="mlops/calibration.json")
    p.add_argument("--out-figure", default="mlops/figures/ambiguity_calibration.png")
    args = p.parse_args()

    # ── ClearML Task: tracks the calibration as an artifact and figure ──────
    task = Task.init(
        project_name="Sen1Floods11/Calibration",
        task_name="empirical-ambiguity-band",
        task_type=Task.TaskTypes.data_processing,
    )
    task.connect(vars(args), name="calibration_args")

    bin_edges = np.arange(0.0, args.max_distance_db + args.bin_width_db,
                          args.bin_width_db)

    stats = calibrate(
        s1_dir=Path(args.s1_dir),
        label_dir=Path(args.label_dir),
        split_csv=Path(args.split_csv),
        db_threshold=args.db_threshold,
        bin_edges=bin_edges,
        target_accuracy=args.target_accuracy,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nEmpirical ambiguity band: ±{stats['band_edge_db']:.2f} dB")
    print(f"  → cascade routes pixels with VV ∈ "
          f"[{args.db_threshold - stats['band_edge_db']:.2f}, "
          f"{args.db_threshold + stats['band_edge_db']:.2f}] dB to deep model")
    print(f"Wrote calibration → {out_json}")

    make_figure(stats, Path(args.out_figure))

    # Register both as ClearML artifacts so the cascade pipeline can pull
    # them by Task ID without dragging filesystem paths around.
    task.upload_artifact(name="calibration", artifact_object=str(out_json))
    task.upload_artifact(name="figure_a",    artifact_object=str(args.out_figure))
    task.get_logger().report_scalar(
        title="calibration", series="band_edge_db",
        value=stats["band_edge_db"], iteration=0,
    )
    print(f"\nClearML Task ID: {task.id}")


if __name__ == "__main__":
    main()
