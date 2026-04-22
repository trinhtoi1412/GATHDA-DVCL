from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.hda.experiment import main


PROJECT_ROOT = Path(__file__).resolve().parent
output_dir = PROJECT_ROOT / "outputs" / "proposed_reference"
output_dir.mkdir(parents=True, exist_ok=True)

tuning_dir = PROJECT_ROOT / "outputs" / "tuning_proposed"
best_candidates = [
    tuning_dir / "best_config_sequential.json",
    tuning_dir / "best_config.json",
]

best_path = None
for p in best_candidates:
    if p.exists():
        best_path = p
        break

if best_path is None:
    raise FileNotFoundError(f"Cannot find best config in {tuning_dir}")

best_cfg = json.loads(best_path.read_text(encoding="utf-8"))
k_similarity = int(best_cfg["k_similarity"])
contrastive_weight = float(best_cfg["contrastive_weight"])
dual_view_weight = float(best_cfg["dual_view_weight"])

print("Using tuned best params from:", best_path)
print(
    {
        "k_similarity": k_similarity,
        "contrastive_weight": contrastive_weight,
        "dual_view_weight": dual_view_weight,
    }
)


def interpolate_mean_curve(x_values, y_values, grid):
    curves = []
    for x, y in zip(x_values, y_values):
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        x_unique, idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[idx]
        curves.append(np.interp(grid, x_unique, y_unique))
    return np.mean(np.vstack(curves), axis=0)

result = main(
    use_extra_similarity=True,
    use_contrastive=True,
    use_dual_view_cl=True,
    use_attention_fusion=True,
    k_similarity=k_similarity,
    n_folds=5,
    max_positive_edges=300000,
    max_epochs=500,
    batch_size=24576,
    graph_pos_edge_cap=300000,
    patience=30,
    contrastive_temperature=0.2,
    contrastive_weight=contrastive_weight,
    contrastive_max_pos=1536,
    dual_view_temperature=0.2,
    dual_view_weight=dual_view_weight,
    dual_view_max_nodes=6144,
    curve_output_prefix=output_dir / "proposed_best_curves",
    curve_dpi=300,
)

summary_df = pd.DataFrame([result["summary"]])
summary_df.to_csv(output_dir / "proposed_summary.csv", index=False)
result["fold_metrics"].to_csv(output_dir / "proposed_fold_metrics.csv", index=False)

np.savez_compressed(
    output_dir / "proposed_predictions_all.npz",
    y_true=result["y_true"],
    y_score=result["y_score"],
)

curve_archive = {}
curve_rows = []
for fold_curve in result["fold_curves"]:
    fold = int(fold_curve["fold"])
    curve_archive[f"roc_fpr_fold{fold}"] = fold_curve["roc_fpr"]
    curve_archive[f"roc_tpr_fold{fold}"] = fold_curve["roc_tpr"]
    curve_archive[f"pr_rec_fold{fold}"] = fold_curve["pr_rec"]
    curve_archive[f"pr_pre_fold{fold}"] = fold_curve["pr_pre"]

    for i in range(len(fold_curve["roc_fpr"])):
        curve_rows.append(
            {
                "fold": fold,
                "curve": "roc",
                "point_idx": i,
                "x": float(fold_curve["roc_fpr"][i]),
                "y": float(fold_curve["roc_tpr"][i]),
                "fold_auc": float(fold_curve["auc"]),
                "fold_aupr": float(fold_curve["aupr"]),
            }
        )

    for i in range(len(fold_curve["pr_rec"])):
        curve_rows.append(
            {
                "fold": fold,
                "curve": "pr",
                "point_idx": i,
                "x": float(fold_curve["pr_rec"][i]),
                "y": float(fold_curve["pr_pre"][i]),
                "fold_auc": float(fold_curve["auc"]),
                "fold_aupr": float(fold_curve["aupr"]),
            }
        )

np.savez_compressed(output_dir / "proposed_curve_arrays_per_fold.npz", **curve_archive)
pd.DataFrame(curve_rows).to_csv(output_dir / "proposed_curve_points_per_fold.csv", index=False)

roc_grid = np.linspace(0.0, 1.0, 400)
roc_mean = interpolate_mean_curve(
    [fold_curve["roc_fpr"] for fold_curve in result["fold_curves"]],
    [fold_curve["roc_tpr"] for fold_curve in result["fold_curves"]],
    roc_grid,
)

pr_grid = np.linspace(0.0, 1.0, 400)
pr_mean = interpolate_mean_curve(
    [fold_curve["pr_rec"] for fold_curve in result["fold_curves"]],
    [fold_curve["pr_pre"] for fold_curve in result["fold_curves"]],
    pr_grid,
)

mean_rows = []
for i in range(len(roc_grid)):
    mean_rows.append(
        {
            "curve": "roc",
            "point_idx": i,
            "x": float(roc_grid[i]),
            "y": float(roc_mean[i]),
            "metric_mean": float(result["metric_mean"]["auc"]),
            "metric_std": float(result["metric_std"]["auc"]),
        }
    )

for i in range(len(pr_grid)):
    mean_rows.append(
        {
            "curve": "pr",
            "point_idx": i,
            "x": float(pr_grid[i]),
            "y": float(pr_mean[i]),
            "metric_mean": float(result["metric_mean"]["aupr"]),
            "metric_std": float(result["metric_std"]["aupr"]),
        }
    )

pd.DataFrame(mean_rows).to_csv(output_dir / "proposed_curve_points_mean.csv", index=False)

pd.DataFrame(
    [
        {
            "best_config_source": str(best_path),
            "k_similarity": k_similarity,
            "contrastive_weight": contrastive_weight,
            "dual_view_weight": dual_view_weight,
            "n_folds": 5,
            "max_epochs": 500,
            "batch_size": 24576,
            "graph_pos_edge_cap": 300000,
            "max_positive_edges": 300000,
        }
    ]
).to_csv(output_dir / "proposed_used_params.csv", index=False)

curve_bundle_meta = {
    "best_config_source": str(best_path),
    "k_similarity": k_similarity,
    "contrastive_weight": contrastive_weight,
    "dual_view_weight": dual_view_weight,
    "n_folds": 5,
    "curve_files": {
        "curve_image_png": str(output_dir / "proposed_best_curves.png"),
        "curve_image_jpg": str(output_dir / "proposed_best_curves.jpg"),
        "predictions_all": str(output_dir / "proposed_predictions_all.npz"),
        "curve_arrays_per_fold": str(output_dir / "proposed_curve_arrays_per_fold.npz"),
        "curve_points_per_fold": str(output_dir / "proposed_curve_points_per_fold.csv"),
        "curve_points_mean": str(output_dir / "proposed_curve_points_mean.csv"),
        "fold_metrics": str(output_dir / "proposed_fold_metrics.csv"),
        "summary": str(output_dir / "proposed_summary.csv"),
        "used_params": str(output_dir / "proposed_used_params.csv"),
    },
    "metric_mean": result["metric_mean"],
    "metric_std": result["metric_std"],
}
(output_dir / "proposed_curve_bundle_metadata.json").write_text(
    json.dumps(curve_bundle_meta, indent=2),
    encoding="utf-8",
)

print(summary_df)
print("Saved full curve data bundle to", output_dir)
