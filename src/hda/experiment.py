import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, train_test_split
from sympy import false

from .config import ExperimentConfig
from .data import (
    build_adjacency_edge_index,
    build_edge_index,
    build_similarity_edge_index,
    collect_eval_edges,
    load_core_data,
    prepare_knn_graphs,
    set_seed,
)
from .model import (
    GATMDAScalable,
    contrastive_pair_loss,
    dual_view_contrastive_loss,
    score_edge_batchwise,
)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int32)
    out = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "aupr": float(average_precision_score(y_true, y_score)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["auc"] = float("nan")
    return out


def train_single_fold(
    train_R: np.ndarray,
    train_edges: np.ndarray,
    train_labels: np.ndarray,
    val_edges: np.ndarray,
    val_labels: np.ndarray,
    test_edges: np.ndarray,
    herb_idx: np.ndarray,
    herb_w: np.ndarray,
    dis_idx: np.ndarray,
    dis_w: np.ndarray,
    use_contrastive: bool,
    use_dual_view_cl: bool,
    use_attention_fusion: bool,
    cfg: ExperimentConfig,
    fold_seed: int,
) -> Tuple[np.ndarray, int, float]:
    device = torch.device(cfg.device)
    n_herb, n_dis = train_R.shape

    model = GATMDAScalable(n_herb, n_dis, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    amp_enabled = false
    scaler = None

    # scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    single_edge_index, single_edge_weight = build_edge_index(
        train_R,
        herb_idx,
        herb_w,
        dis_idx,
        dis_w,
        graph_pos_edge_cap=cfg.graph_pos_edge_cap,
        seed=fold_seed,
    )
    adj_edge_index, adj_edge_weight = build_adjacency_edge_index(
        train_R,
        graph_pos_edge_cap=cfg.graph_pos_edge_cap,
        seed=fold_seed,
    )
    sim_edge_index, sim_edge_weight = build_similarity_edge_index(
        herb_idx,
        herb_w,
        dis_idx,
        dis_w,
        n_herb=n_herb,
        n_dis=n_dis,
    )

    use_dual_graph = bool(use_dual_view_cl or use_attention_fusion)
    if use_dual_graph:
        edge_index_adj = adj_edge_index.to(device)
        edge_weight_adj = adj_edge_weight.to(device)
        edge_index_sim = sim_edge_index.to(device)
        edge_weight_sim = sim_edge_weight.to(device)
    else:
        edge_index_adj = single_edge_index.to(device)
        edge_weight_adj = single_edge_weight.to(device)
        edge_index_sim = None
        edge_weight_sim = None

    h_train = (train_edges // n_dis).astype(np.int64)
    d_train = (train_edges % n_dis).astype(np.int64)

    order = np.arange(len(train_edges))
    rng = np.random.default_rng(fold_seed)

    best_state = None
    best_val_aupr = -1.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, cfg.max_epochs + 1):
        rng.shuffle(order)
        model.train()

        for st in range(0, len(order), cfg.batch_size):
            ed = min(st + cfg.batch_size, len(order))
            b = order[st:ed]

            hb_np = h_train[b]
            db_np = d_train[b]
            yb_np = train_labels[b]

            hb = torch.tensor(hb_np, dtype=torch.long, device=device)
            db = torch.tensor(db_np, dtype=torch.long, device=device)
            yb = torch.tensor(yb_np, dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            try:
                if amp_enabled:
                    with torch.amp.autocast("cuda", enabled=True):
                        herb_z, dis_z, aux = model.encode(
                            edge_index_adj=edge_index_adj,
                            edge_weight_adj=edge_weight_adj,
                            edge_index_sim=edge_index_sim,
                            edge_weight_sim=edge_weight_sim,
                            use_attention_fusion=use_attention_fusion,
                        )
                        logit = model.score_pairs(herb_z, dis_z, hb, db)
                        loss = F.binary_cross_entropy_with_logits(logit, yb)

                        if use_contrastive:
                            loss = loss + cfg.contrastive_weight * contrastive_pair_loss(
                                herb_z,
                                dis_z,
                                hb,
                                db,
                                yb,
                                temperature=cfg.contrastive_temperature,
                                max_pos=cfg.contrastive_max_pos,
                            )

                        if use_dual_view_cl:
                            loss = loss + cfg.dual_view_weight * dual_view_contrastive_loss(
                                aux["z_adj"],
                                aux["z_sim"],
                                temperature=cfg.dual_view_temperature,
                                max_nodes=cfg.dual_view_max_nodes,
                            )
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    herb_z, dis_z, aux = model.encode(
                        edge_index_adj=edge_index_adj,
                        edge_weight_adj=edge_weight_adj,
                        edge_index_sim=edge_index_sim,
                        edge_weight_sim=edge_weight_sim,
                        use_attention_fusion=use_attention_fusion,
                    )
                    logit = model.score_pairs(herb_z, dis_z, hb, db)
                    loss = F.binary_cross_entropy_with_logits(logit, yb)

                    if use_contrastive:
                        loss = loss + cfg.contrastive_weight * contrastive_pair_loss(
                            herb_z,
                            dis_z,
                            hb,
                            db,
                            yb,
                            temperature=cfg.contrastive_temperature,
                            max_pos=cfg.contrastive_max_pos,
                        )

                    if use_dual_view_cl:
                        loss = loss + cfg.dual_view_weight * dual_view_contrastive_loss(
                            aux["z_adj"],
                            aux["z_sim"],
                            temperature=cfg.dual_view_temperature,
                            max_nodes=cfg.dual_view_max_nodes,
                        )
                    loss.backward()
                    opt.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device.type == "cuda":
                    torch.cuda.empty_cache()
                raise

        val_scores = score_edge_batchwise(
            model=model,
            edge_index_adj=edge_index_adj,
            edge_weight_adj=edge_weight_adj,
            edge_index_sim=edge_index_sim,
            edge_weight_sim=edge_weight_sim,
            use_attention_fusion=use_attention_fusion,
            eval_edges=val_edges,
            n_dis=n_dis,
            device=device,
            batch_size=cfg.batch_size,
        )
        val_metrics = compute_metrics(val_labels, val_scores, threshold=cfg.decision_threshold)
        val_aupr = val_metrics["aupr"]

        improved = val_aupr > (best_val_aupr + cfg.early_stopping_min_delta)
        if improved:
            best_val_aupr = val_aupr
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_scores = score_edge_batchwise(
        model=model,
        edge_index_adj=edge_index_adj,
        edge_weight_adj=edge_weight_adj,
        edge_index_sim=edge_index_sim,
        edge_weight_sim=edge_weight_sim,
        use_attention_fusion=use_attention_fusion,
        eval_edges=test_edges,
        n_dis=n_dis,
        device=device,
        batch_size=cfg.batch_size,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return test_scores, best_epoch, best_val_aupr


def run_experiment(
    use_extra_similarity: bool,
    use_contrastive: bool,
    use_dual_view_cl: bool,
    use_attention_fusion: bool,
    k_similarity: int,
    cfg: ExperimentConfig,
):
    set_seed(cfg.random_seed)
    data_bundle = load_core_data(cfg.data_root)
    R = data_bundle["R"].astype(np.float32)

    eval_edges, eval_labels = collect_eval_edges(R, cfg)
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_seed)

    all_true: List[np.ndarray] = []
    all_score: List[np.ndarray] = []
    fold_metrics: List[Dict[str, float]] = []
    fold_curves: List[Dict[str, object]] = []

    config_name = (
        f"extra={int(use_extra_similarity)}|pair_cl={int(use_contrastive)}|"
        f"dual_cl={int(use_dual_view_cl)}|attn_fuse={int(use_attention_fusion)}|k={k_similarity}"
    )
    print(f"\n=== CONFIG {config_name} ===")

    for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(eval_edges), start=1):
        t0 = time.time()
        fold_seed = cfg.random_seed + fold_id

        trainval_edges = eval_edges[trainval_idx]
        trainval_labels = eval_labels[trainval_idx]
        test_edges = eval_edges[test_idx]
        test_labels = eval_labels[test_idx]

        tr_idx, va_idx = train_test_split(
            np.arange(len(trainval_edges)),
            test_size=cfg.val_ratio_in_trainval,
            random_state=fold_seed,
            shuffle=True,
            stratify=trainval_labels,
        )

        train_edges = trainval_edges[tr_idx]
        train_labels = trainval_labels[tr_idx].astype(np.float32)
        val_edges = trainval_edges[va_idx]
        val_labels = trainval_labels[va_idx]

        train_R = R.copy()
        pos_test_edges = test_edges[test_labels == 1]
        pos_val_edges = val_edges[val_labels == 1]
        train_R.reshape(-1)[pos_test_edges] = 0.0
        train_R.reshape(-1)[pos_val_edges] = 0.0

        h_idx, h_w, d_idx, d_w = prepare_knn_graphs(
            data_bundle,
            train_R,
            k=cfg.topk_similarity,
            use_extra_similarity=use_extra_similarity,
        )

        try:
            test_scores, best_epoch, best_val_aupr = train_single_fold(
                train_R,
                train_edges=train_edges,
                train_labels=train_labels,
                val_edges=val_edges,
                val_labels=val_labels,
                test_edges=test_edges,
                herb_idx=h_idx,
                herb_w=h_w,
                dis_idx=d_idx,
                dis_w=d_w,
                use_contrastive=use_contrastive,
                use_dual_view_cl=use_dual_view_cl,
                use_attention_fusion=use_attention_fusion,
                cfg=cfg,
                fold_seed=fold_seed,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cfg.device.startswith("cuda"):
                print("OOM on GPU, retry fold on CPU with lower batch size")
                cpu_cfg = ExperimentConfig(
                    **{
                        **cfg.__dict__,
                        "device": "cpu",
                        "batch_size": max(4096, cfg.batch_size // 4),
                        "use_amp": False,
                    }
                )
                test_scores, best_epoch, best_val_aupr = train_single_fold(
                    train_R,
                    train_edges=train_edges,
                    train_labels=train_labels,
                    val_edges=val_edges,
                    val_labels=val_labels,
                    test_edges=test_edges,
                    herb_idx=h_idx,
                    herb_w=h_w,
                    dis_idx=d_idx,
                    dis_w=d_w,
                    use_contrastive=use_contrastive,
                    use_dual_view_cl=use_dual_view_cl,
                    use_attention_fusion=use_attention_fusion,
                    cfg=cpu_cfg,
                    fold_seed=fold_seed,
                )
            else:
                raise

        m = compute_metrics(test_labels, test_scores, threshold=cfg.decision_threshold)
        m["fold"] = fold_id
        m["best_epoch"] = best_epoch
        m["best_val_aupr"] = best_val_aupr
        m["seconds"] = time.time() - t0
        fold_metrics.append(m)
        all_true.append(test_labels)
        all_score.append(test_scores)

        roc_fpr, roc_tpr, _ = roc_curve(test_labels, test_scores)
        pr_rec, pr_pre, _ = precision_recall_curve(test_labels, test_scores)
        fold_curves.append(
            {
                "fold": fold_id,
                "roc_fpr": roc_fpr,
                "roc_tpr": roc_tpr,
                "pr_rec": pr_rec,
                "pr_pre": pr_pre,
                "auc": m["auc"],
                "aupr": m["aupr"],
            }
        )

        print(
            f"fold={fold_id} stop_epoch={best_epoch} "
            f"acc={m['acc']:.4f} f1={m['f1']:.4f} precision={m['precision']:.4f} "
            f"recall={m['recall']:.4f} aupr={m['aupr']:.4f} auc={m['auc']:.4f}"
        )

    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)
    summary = compute_metrics(y_true, y_score, threshold=cfg.decision_threshold)
    fold_df = pd.DataFrame(fold_metrics)

    metric_cols = ["acc", "f1", "precision", "recall", "aupr", "auc"]
    metric_mean = fold_df[metric_cols].mean(numeric_only=True)
    metric_std = fold_df[metric_cols].std(numeric_only=True)

    print(f"--- CONFIG SUMMARY {config_name} ---")
    for c in metric_cols:
        print(f"{c}: mean={metric_mean[c]:.4f}, std={metric_std[c]:.4f}")

    return {
        "summary": summary,
        "fold_metrics": fold_df,
        "metric_mean": metric_mean.to_dict(),
        "metric_std": metric_std.to_dict(),
        "fold_curves": fold_curves,
        "y_true": y_true,
        "y_score": y_score,
    }


def _interpolate_mean_curve(x_values: List[np.ndarray], y_values: List[np.ndarray], grid: np.ndarray) -> np.ndarray:
    curves = []
    for x, y in zip(x_values, y_values):
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        x_unique, idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[idx]
        curves.append(np.interp(grid, x_unique, y_unique))
    return np.mean(np.vstack(curves), axis=0)


def plot_curves(
    result: Dict[str, object],
    title: str,
    save_prefix: Optional[Path | str] = None,
    dpi: int = 300,
) -> None:
    fold_curves = result["fold_curves"]
    n_fold = len(fold_curves)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_fold, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, fold_curve in enumerate(fold_curves):
        axes[0].plot(
            fold_curve["roc_fpr"],
            fold_curve["roc_tpr"],
            color=colors[i],
            linewidth=1.4,
            label=f"Fold {fold_curve['fold']} (AUC={fold_curve['auc']:.4f})",
        )

    roc_grid = np.linspace(0.0, 1.0, 400)
    roc_mean = _interpolate_mean_curve(
        [fold_curve["roc_fpr"] for fold_curve in fold_curves],
        [fold_curve["roc_tpr"] for fold_curve in fold_curves],
        roc_grid,
    )
    axes[0].plot(
        roc_grid,
        roc_mean,
        color="black",
        linewidth=2.2,
        label=f"Mean (AUC={result['metric_mean']['auc']:.4f}±{result['metric_std']['auc']:.4f})",
    )
    axes[0].set_title(f"ROC | {title}")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend(loc="lower right")

    for i, fold_curve in enumerate(fold_curves):
        axes[1].plot(
            fold_curve["pr_rec"],
            fold_curve["pr_pre"],
            color=colors[i],
            linewidth=1.4,
            label=f"Fold {fold_curve['fold']} (AUPR={fold_curve['aupr']:.4f})",
        )

    pr_grid = np.linspace(0.0, 1.0, 400)
    pr_mean = _interpolate_mean_curve(
        [fold_curve["pr_rec"] for fold_curve in fold_curves],
        [fold_curve["pr_pre"] for fold_curve in fold_curves],
        pr_grid,
    )
    axes[1].plot(
        pr_grid,
        pr_mean,
        color="black",
        linewidth=2.2,
        label=f"Mean (AUPR={result['metric_mean']['aupr']:.4f}±{result['metric_std']['aupr']:.4f})",
    )
    axes[1].set_title(f"PR | {title}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    if save_prefix is not None:
        prefix = Path(save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        png_path = prefix.with_suffix(".png")
        jpg_path = prefix.with_suffix(".jpg")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(jpg_path, dpi=dpi, bbox_inches="tight", format="jpg")
        print(f"saved curves: {png_path}")
        print(f"saved curves: {jpg_path}")
    plt.show()
    plt.close(fig)


def main(
    use_extra_similarity: bool,
    use_contrastive: bool,
    use_dual_view_cl: bool,
    use_attention_fusion: bool,
    k_similarity: int = 20,
    n_folds: int = 5,
    max_positive_edges: Optional[int] = 300000,
    max_epochs: int = 30,
    batch_size: int = 32768,
    graph_pos_edge_cap: int = 300000,
    patience: int = 5,
    contrastive_temperature: float = 0.2,
    contrastive_weight: float = 0.1,
    contrastive_max_pos: int = 1024,
    dual_view_temperature: float = 0.2,
    dual_view_weight: float = 0.15,
    dual_view_max_nodes: int = 4096,
    curve_output_prefix: Optional[Path | str] = None,
    curve_dpi: int = 300,
):
    cfg = ExperimentConfig(
        n_folds=n_folds,
        max_positive_edges=max_positive_edges,
        topk_similarity=k_similarity,
        max_epochs=max_epochs,
        batch_size=batch_size,
        graph_pos_edge_cap=graph_pos_edge_cap,
        early_stopping_patience=patience,
        contrastive_temperature=contrastive_temperature,
        contrastive_weight=contrastive_weight,
        contrastive_max_pos=contrastive_max_pos,
        dual_view_temperature=dual_view_temperature,
        dual_view_weight=dual_view_weight,
        dual_view_max_nodes=dual_view_max_nodes,
    )

    result = run_experiment(
        use_extra_similarity=use_extra_similarity,
        use_contrastive=use_contrastive,
        use_dual_view_cl=use_dual_view_cl,
        use_attention_fusion=use_attention_fusion,
        k_similarity=k_similarity,
        cfg=cfg,
    )

    title = (
        f"extra={int(use_extra_similarity)} | pair_cl={int(use_contrastive)} | "
        f"dual_cl={int(use_dual_view_cl)} | attn_fuse={int(use_attention_fusion)} | k={k_similarity}"
    )
    plot_curves(result, title, save_prefix=curve_output_prefix, dpi=curve_dpi)
    return result


def run_ablation(
    k_similarity: int = 20,
    n_folds: int = 5,
    max_positive_edges: Optional[int] = 300000,
    max_epochs: int = 30,
    batch_size: int = 32768,
    graph_pos_edge_cap: int = 300000,
    patience: int = 5,
    contrastive_temperature: float = 0.2,
    contrastive_weight: float = 0.1,
    contrastive_max_pos: int = 1024,
    dual_view_temperature: float = 0.2,
    dual_view_weight: float = 0.15,
    dual_view_max_nodes: int = 4096,
    curve_output_dir: Optional[Path | str] = None,
    curve_dpi: int = 300,
):
    cases = [
        {
            "case": "baseline",
            "use_extra_similarity": False,
            "use_contrastive": False,
            "use_dual_view_cl": False,
            "use_attention_fusion": False,
        },
        {
            "case": "extra_only",
            "use_extra_similarity": True,
            "use_contrastive": False,
            "use_dual_view_cl": False,
            "use_attention_fusion": False,
        },
        {
            "case": "dual_view_only",
            "use_extra_similarity": True,
            "use_contrastive": False,
            "use_dual_view_cl": True,
            "use_attention_fusion": False,
        },
        {
            "case": "attention_only",
            "use_extra_similarity": True,
            "use_contrastive": False,
            "use_dual_view_cl": False,
            "use_attention_fusion": True,
        },
        {
            "case": "full_proposed",
            "use_extra_similarity": True,
            "use_contrastive": True,
            "use_dual_view_cl": True,
            "use_attention_fusion": True,
        },
    ]

    records = []
    outputs = {}
    for cfg_case in cases:
        curve_prefix = None
        if curve_output_dir is not None:
            curve_prefix = Path(curve_output_dir) / f"curves_{cfg_case['case']}"

        res = main(
            use_extra_similarity=cfg_case["use_extra_similarity"],
            use_contrastive=cfg_case["use_contrastive"],
            use_dual_view_cl=cfg_case["use_dual_view_cl"],
            use_attention_fusion=cfg_case["use_attention_fusion"],
            k_similarity=k_similarity,
            n_folds=n_folds,
            max_positive_edges=max_positive_edges,
            max_epochs=max_epochs,
            batch_size=batch_size,
            graph_pos_edge_cap=graph_pos_edge_cap,
            patience=patience,
            contrastive_temperature=contrastive_temperature,
            contrastive_weight=contrastive_weight,
            contrastive_max_pos=contrastive_max_pos,
            dual_view_temperature=dual_view_temperature,
            dual_view_weight=dual_view_weight,
            dual_view_max_nodes=dual_view_max_nodes,
            curve_output_prefix=curve_prefix,
            curve_dpi=curve_dpi,
        )

        case_name = cfg_case["case"]
        outputs[case_name] = res
        row = {
            "case": case_name,
            "use_extra_similarity": int(cfg_case["use_extra_similarity"]),
            "use_contrastive": int(cfg_case["use_contrastive"]),
            "use_dual_view_cl": int(cfg_case["use_dual_view_cl"]),
            "use_attention_fusion": int(cfg_case["use_attention_fusion"]),
            "k_similarity": k_similarity,
        }
        row.update(res["summary"])
        records.append(row)

    table = pd.DataFrame(records)
    return table, outputs
