import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.hda.config import ExperimentConfig
from src.hda.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--max-positive-edges", type=int, default=300000)
    parser.add_argument("--graph-pos-edge-cap", type=int, default=300000)
    return parser.parse_args()


def is_better(candidate: Dict[str, float], best: Dict[str, float]) -> bool:
    cand_key: Tuple[float, float] = (candidate["mean_aupr"], candidate["mean_auc"])
    best_key: Tuple[float, float] = (best["mean_aupr"], best["mean_auc"])
    return cand_key > best_key


def run_one(
    stage: str,
    trial_name: str,
    k_similarity: int,
    contrastive_weight: float,
    dual_view_weight: float,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, float]:
    cfg = ExperimentConfig(
        n_folds=args.n_folds,
        max_positive_edges=args.max_positive_edges,
        topk_similarity=k_similarity,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        graph_pos_edge_cap=args.graph_pos_edge_cap,
        early_stopping_patience=args.patience,
        contrastive_temperature=0.2,
        contrastive_weight=contrastive_weight,
        contrastive_max_pos=1536,
        dual_view_temperature=0.2,
        dual_view_weight=dual_view_weight,
        dual_view_max_nodes=6144,
    )

    print(
        f"[{stage}] {trial_name} | k={k_similarity} "
        f"contrastive_weight={contrastive_weight} dual_view_weight={dual_view_weight}"
    )

    result = run_experiment(
        use_extra_similarity=True,
        use_contrastive=True,
        use_dual_view_cl=True,
        use_attention_fusion=True,
        k_similarity=k_similarity,
        cfg=cfg,
    )

    fold_file = output_dir / f"fold_metrics_{trial_name}.csv"
    result["fold_metrics"].to_csv(fold_file, index=False)

    row = {
        "stage": stage,
        "trial_name": trial_name,
        "k_similarity": k_similarity,
        "contrastive_weight": contrastive_weight,
        "dual_view_weight": dual_view_weight,
        "summary_acc": result["summary"]["acc"],
        "summary_f1": result["summary"]["f1"],
        "summary_precision": result["summary"]["precision"],
        "summary_recall": result["summary"]["recall"],
        "summary_aupr": result["summary"]["aupr"],
        "summary_auc": result["summary"]["auc"],
        "mean_acc": result["metric_mean"]["acc"],
        "mean_f1": result["metric_mean"]["f1"],
        "mean_precision": result["metric_mean"]["precision"],
        "mean_recall": result["metric_mean"]["recall"],
        "mean_aupr": result["metric_mean"]["aupr"],
        "mean_auc": result["metric_mean"]["auc"],
        "std_aupr": result["metric_std"]["aupr"],
        "std_auc": result["metric_std"]["auc"],
    }
    return row


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "outputs" / "tuning_proposed"
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = [15, 20, 25, 30, 35, 40]
    contrastive_weight_values = [0.08, 0.16]
    dual_view_weight_values = [0.30]

    base = {
        "k_similarity": 25,
        "contrastive_weight": 0.12,
        "dual_view_weight": 0.22,
    }

    total_cases = len(k_values) + len(contrastive_weight_values) + len(dual_view_weight_values)
    print(f"planned_cases={total_cases}")

    records = []
    run_idx = 0

    best = None
    for k_val in k_values:
        run_idx += 1
        trial_name = f"{run_idx:02d}_k{k_val}_cw{base['contrastive_weight']:.2f}_dw{base['dual_view_weight']:.2f}"
        row = run_one(
            stage="k_similarity",
            trial_name=trial_name,
            k_similarity=k_val,
            contrastive_weight=base["contrastive_weight"],
            dual_view_weight=base["dual_view_weight"],
            args=args,
            output_dir=output_dir,
        )
        records.append(row)
        if best is None or is_better(row, best):
            best = row
            base["k_similarity"] = k_val

    for cw_val in contrastive_weight_values:
        run_idx += 1
        trial_name = f"{run_idx:02d}_k{base['k_similarity']}_cw{cw_val:.2f}_dw{base['dual_view_weight']:.2f}"
        row = run_one(
            stage="contrastive_weight",
            trial_name=trial_name,
            k_similarity=base["k_similarity"],
            contrastive_weight=cw_val,
            dual_view_weight=base["dual_view_weight"],
            args=args,
            output_dir=output_dir,
        )
        records.append(row)
        if is_better(row, best):
            best = row
            base["contrastive_weight"] = cw_val

    for dw_val in dual_view_weight_values:
        run_idx += 1
        trial_name = f"{run_idx:02d}_k{base['k_similarity']}_cw{base['contrastive_weight']:.2f}_dw{dw_val:.2f}"
        row = run_one(
            stage="dual_view_weight",
            trial_name=trial_name,
            k_similarity=base["k_similarity"],
            contrastive_weight=base["contrastive_weight"],
            dual_view_weight=dw_val,
            args=args,
            output_dir=output_dir,
        )
        records.append(row)
        if is_better(row, best):
            best = row
            base["dual_view_weight"] = dw_val

    tuning_table = pd.DataFrame(records)
    tuning_table.to_csv(output_dir / "tuning_summary_sequential.csv", index=False)

    ranked = tuning_table.sort_values(["mean_aupr", "mean_auc"], ascending=False).reset_index(drop=True)
    ranked.to_csv(output_dir / "tuning_ranked_sequential.csv", index=False)

    final_best = {
        "k_similarity": base["k_similarity"],
        "contrastive_weight": base["contrastive_weight"],
        "dual_view_weight": base["dual_view_weight"],
        "best_mean_aupr": best["mean_aupr"],
        "best_mean_auc": best["mean_auc"],
        "planned_cases": total_cases,
        "executed_cases": int(len(records)),
    }
    (output_dir / "best_config_sequential.json").write_text(json.dumps(final_best, indent=2), encoding="utf-8")

    print(ranked[["stage", "trial_name", "k_similarity", "contrastive_weight", "dual_view_weight", "mean_aupr", "mean_auc"]])


if __name__ == "__main__":
    main()
