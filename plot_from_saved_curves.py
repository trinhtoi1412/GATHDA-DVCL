from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_fold_ids(curve_archive: np.lib.npyio.NpzFile) -> list[int]:
    fold_ids = []
    for key in curve_archive.files:
        if key.startswith("roc_fpr_fold"):
            fold_ids.append(int(key.replace("roc_fpr_fold", "")))
    return sorted(fold_ids)


def plot_saved_curves(
    data_dir: Path,
    output_prefix: str,
    dpi: int,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    npz_path = data_dir / "proposed_curve_arrays_per_fold.npz"
    metrics_path = data_dir / "proposed_fold_metrics.csv"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing curve archive: {npz_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing fold metrics: {metrics_path}")

    curve_archive = np.load(npz_path)
    fold_metrics = pd.read_csv(metrics_path)

    fold_ids = load_fold_ids(curve_archive)
    if len(fold_ids) == 0:
        raise RuntimeError("No fold curves found in archive")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fold_ids), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    roc_fprs = []
    roc_tprs = []
    pr_recs = []
    pr_pres = []

    for i, fold in enumerate(fold_ids):
        roc_fpr = curve_archive[f"roc_fpr_fold{fold}"]
        roc_tpr = curve_archive[f"roc_tpr_fold{fold}"]
        pr_rec = curve_archive[f"pr_rec_fold{fold}"]
        pr_pre = curve_archive[f"pr_pre_fold{fold}"]

        roc_fprs.append(roc_fpr)
        roc_tprs.append(roc_tpr)
        pr_recs.append(pr_rec)
        pr_pres.append(pr_pre)

        fold_row = fold_metrics.loc[fold_metrics["fold"] == fold].iloc[0]
        fold_auc = float(fold_row["auc"])
        fold_aupr = float(fold_row["aupr"])

        axes[0].plot(
            roc_fpr,
            roc_tpr,
            color=colors[i],
            linewidth=1.8,
            alpha=0.95,
            label=f"Fold {fold} (AUC={fold_auc:.4f})",
            zorder=3,
        )
        axes[1].plot(
            pr_rec,
            pr_pre,
            color=colors[i],
            linewidth=1.8,
            alpha=0.95,
            label=f"Fold {fold} (AUPR={fold_aupr:.4f})",
            zorder=3,
        )

    roc_grid = np.linspace(0.0, 1.0, 400)
    roc_interp = []
    for x, y in zip(roc_fprs, roc_tprs):
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        x_unique, idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[idx]
        roc_interp.append(np.interp(roc_grid, x_unique, y_unique))
    roc_mean = np.mean(np.vstack(roc_interp), axis=0)

    pr_grid = np.linspace(0.0, 1.0, 400)
    pr_interp = []
    for x, y in zip(pr_recs, pr_pres):
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        x_unique, idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[idx]
        pr_interp.append(np.interp(pr_grid, x_unique, y_unique))
    pr_mean = np.mean(np.vstack(pr_interp), axis=0)

    auc_mean = float(fold_metrics["auc"].mean())
    auc_std = float(fold_metrics["auc"].std())
    aupr_mean = float(fold_metrics["aupr"].mean())
    aupr_std = float(fold_metrics["aupr"].std())

    mean_color = "#C2185B"
    axes[0].plot(
        roc_grid,
        roc_mean,
        color=mean_color,
        linewidth=1.2,
        linestyle="--",
        alpha=0.95,
        label=f"Mean (AUC={auc_mean:.4f}±{auc_std:.4f})",
        zorder=2,
    )
    axes[1].plot(
        pr_grid,
        pr_mean,
        color=mean_color,
        linewidth=1.2,
        linestyle="--",
        alpha=0.95,
        label=f"Mean (AUPR={aupr_mean:.4f}±{aupr_std:.4f})",
        zorder=2,
    )

    axes[0].set_title("ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend(loc="lower right")

    axes[1].set_title("PR")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()

    output_base = data_dir / output_prefix
    output_png = output_base.with_suffix(".png")
    output_jpg = output_base.with_suffix(".jpg")

    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_jpg, dpi=dpi, bbox_inches="tight", format="jpg")
    plt.close(fig)

    roc_mean_fig, roc_ax = plt.subplots(figsize=(6.5, 5))
    roc_ax.plot(
        roc_grid,
        roc_mean,
        color=mean_color,
        linewidth=1.8,
        linestyle="-",
        alpha=0.98,
        label=f"Mean (AUC={auc_mean:.4f}±{auc_std:.4f})",
    )
    roc_ax.set_title("ROC")
    roc_ax.set_xlabel("FPR")
    roc_ax.set_ylabel("TPR")
    roc_ax.legend(loc="lower right")
    roc_mean_fig.tight_layout()

    roc_mean_png = data_dir / f"{output_prefix}_roc_mean_only.png"
    roc_mean_jpg = data_dir / f"{output_prefix}_roc_mean_only.jpg"
    roc_mean_fig.savefig(roc_mean_png, dpi=dpi, bbox_inches="tight")
    roc_mean_fig.savefig(roc_mean_jpg, dpi=dpi, bbox_inches="tight", format="jpg")
    plt.close(roc_mean_fig)

    pr_mean_fig, pr_ax = plt.subplots(figsize=(6.5, 5))
    pr_ax.plot(
        pr_grid,
        pr_mean,
        color=mean_color,
        linewidth=1.8,
        linestyle="-",
        alpha=0.98,
        label=f"Mean (AUPR={aupr_mean:.4f}±{aupr_std:.4f})",
    )
    pr_ax.set_title("PR")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.legend(loc="lower left")
    pr_mean_fig.tight_layout()

    pr_mean_png = data_dir / f"{output_prefix}_pr_mean_only.png"
    pr_mean_jpg = data_dir / f"{output_prefix}_pr_mean_only.jpg"
    pr_mean_fig.savefig(pr_mean_png, dpi=dpi, bbox_inches="tight")
    pr_mean_fig.savefig(pr_mean_jpg, dpi=dpi, bbox_inches="tight", format="jpg")
    plt.close(pr_mean_fig)

    return output_png, output_jpg, roc_mean_png, roc_mean_jpg, pr_mean_png, pr_mean_jpg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/proposed_reference"))
    parser.add_argument("--output-prefix", type=str, default="proposed_best_curves_restyled")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_png, out_jpg, roc_mean_png, roc_mean_jpg, pr_mean_png, pr_mean_jpg = plot_saved_curves(
        args.data_dir,
        args.output_prefix,
        args.dpi,
    )
    print("saved", out_png)
    print("saved", out_jpg)
    print("saved", roc_mean_png)
    print("saved", roc_mean_jpg)
    print("saved", pr_mean_png)
    print("saved", pr_mean_jpg)


if __name__ == "__main__":
    main()
