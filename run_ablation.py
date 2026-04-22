from pathlib import Path

import numpy as np
import pandas as pd

from src.hda.experiment import run_ablation


PROJECT_ROOT = Path(__file__).resolve().parent


output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

ablation_table, ablation_outputs = run_ablation(
    k_similarity=25,
    n_folds=5,
    max_positive_edges=300000,
    max_epochs=500,
    batch_size=24576,
    graph_pos_edge_cap=300000,
    patience=30,
    contrastive_temperature=0.2,
    contrastive_weight=0.12,
    contrastive_max_pos=1536,
    dual_view_temperature=0.2,
    dual_view_weight=0.22,
    dual_view_max_nodes=6144,
    curve_output_dir=output_dir,
    curve_dpi=300,
)

case_order = [
    "baseline",
    "extra_only",
    "dual_view_only",
    "attention_only",
    "full_proposed",
]
ablation_table["case"] = pd.Categorical(ablation_table["case"], categories=case_order, ordered=True)
ablation_table = ablation_table.sort_values("case").reset_index(drop=True)
ablation_table.to_csv(output_dir / "ablation_summary.csv", index=False)

for case, res in ablation_outputs.items():
    res["fold_metrics"].to_csv(output_dir / f"fold_metrics_{case}.csv", index=False)
    np.savez_compressed(
        output_dir / f"predictions_{case}.npz",
        y_true=res["y_true"],
        y_score=res["y_score"],
    )

print(
    ablation_table[
        [
            "case",
            "use_extra_similarity",
            "use_contrastive",
            "use_dual_view_cl",
            "use_attention_fusion",
            "k_similarity",
            "acc",
            "f1",
            "precision",
            "recall",
            "aupr",
            "auc",
        ]
    ]
)
