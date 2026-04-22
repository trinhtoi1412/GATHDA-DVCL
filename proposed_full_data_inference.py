from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.hda.config import ExperimentConfig
from src.hda.data import (
    build_adjacency_edge_index,
    build_similarity_edge_index,
    load_core_data,
    prepare_knn_graphs,
    set_seed,
)
from src.hda.model import GATMDAScalable, contrastive_pair_loss, dual_view_contrastive_loss


def pick_name_col(df: pd.DataFrame, candidates: list[str], fallback: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return fallback


def main() -> None:
    cfg = ExperimentConfig(
        n_folds=1,
        topk_similarity=25,
        max_positive_edges=300000,
        max_epochs=500,
        batch_size=24576,
        graph_pos_edge_cap=300000,
        early_stopping_patience=30,
        contrastive_temperature=0.2,
        contrastive_weight=0.12,
        contrastive_max_pos=1536,
        dual_view_temperature=0.2,
        dual_view_weight=0.22,
        dual_view_max_nodes=6144,
    )

    output_dir = Path("outputs/proposed_full_data_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.random_seed)
    device = torch.device(cfg.device)
    print("device =", device)

    data_bundle = load_core_data(cfg.data_root)
    R = data_bundle["R"].astype(np.float32)
    n_herb, n_dis = R.shape

    flat = R.reshape(-1)
    pos_idx = np.flatnonzero(flat == 1)
    neg_idx = np.flatnonzero(flat == 0)
    rng = np.random.default_rng(cfg.random_seed)
    neg_sample_size = min(len(pos_idx), len(neg_idx))
    neg_train_idx = rng.choice(neg_idx, size=neg_sample_size, replace=False)

    train_edges = np.concatenate([pos_idx, neg_train_idx])
    train_labels = np.concatenate(
        [
            np.ones(len(pos_idx), dtype=np.float32),
            np.zeros(len(neg_train_idx), dtype=np.float32),
        ]
    )

    h_train = (train_edges // n_dis).astype(np.int64)
    d_train = (train_edges % n_dis).astype(np.int64)

    h_idx, h_w, d_idx, d_w = prepare_knn_graphs(
        data_bundle=data_bundle,
        R_train=R,
        k=cfg.topk_similarity,
        use_extra_similarity=True,
    )

    adj_edge_index, adj_edge_weight = build_adjacency_edge_index(
        R=R,
        graph_pos_edge_cap=cfg.graph_pos_edge_cap,
        seed=cfg.random_seed,
    )
    sim_edge_index, sim_edge_weight = build_similarity_edge_index(
        herb_idx=h_idx,
        herb_w=h_w,
        dis_idx=d_idx,
        dis_w=d_w,
        n_herb=n_herb,
        n_dis=n_dis,
    )

    edge_index_adj = adj_edge_index.to(device)
    edge_weight_adj = adj_edge_weight.to(device)
    edge_index_sim = sim_edge_index.to(device)
    edge_weight_sim = sim_edge_weight.to(device)

    model = GATMDAScalable(n_herb=n_herb, n_dis=n_dis, cfg=cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print("R shape =", R.shape)
    print("positive edges =", len(pos_idx))
    print("sampled negative edges for training =", len(neg_train_idx))

    order = np.arange(len(train_edges))

    for epoch in range(1, cfg.max_epochs + 1):
        rng.shuffle(order)
        model.train()
        epoch_loss = 0.0

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

            if amp_enabled:
                with torch.amp.autocast("cuda", enabled=True):
                    herb_z, dis_z, aux = model.encode(
                        edge_index_adj=edge_index_adj,
                        edge_weight_adj=edge_weight_adj,
                        edge_index_sim=edge_index_sim,
                        edge_weight_sim=edge_weight_sim,
                        use_attention_fusion=True,
                    )
                    logit = model.score_pairs(herb_z, dis_z, hb, db)
                    loss = F.binary_cross_entropy_with_logits(logit, yb)
                    loss = loss + cfg.contrastive_weight * contrastive_pair_loss(
                        herb_z,
                        dis_z,
                        hb,
                        db,
                        yb,
                        temperature=cfg.contrastive_temperature,
                        max_pos=cfg.contrastive_max_pos,
                    )
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
                    use_attention_fusion=True,
                )
                logit = model.score_pairs(herb_z, dis_z, hb, db)
                loss = F.binary_cross_entropy_with_logits(logit, yb)
                loss = loss + cfg.contrastive_weight * contrastive_pair_loss(
                    herb_z,
                    dis_z,
                    hb,
                    db,
                    yb,
                    temperature=cfg.contrastive_temperature,
                    max_pos=cfg.contrastive_max_pos,
                )
                loss = loss + cfg.dual_view_weight * dual_view_contrastive_loss(
                    aux["z_adj"],
                    aux["z_sim"],
                    temperature=cfg.dual_view_temperature,
                    max_nodes=cfg.dual_view_max_nodes,
                )
                loss.backward()
                opt.step()

            epoch_loss += float(loss.detach().cpu())

        print(f"epoch={epoch}/{cfg.max_epochs} loss={epoch_loss:.4f}")

    model_path = output_dir / "proposed_full_data_model.pt"
    torch.save(model.state_dict(), model_path)
    print("saved model to", model_path)

    model.eval()
    with torch.no_grad():
        herb_z, dis_z, _ = model.encode(
            edge_index_adj=edge_index_adj,
            edge_weight_adj=edge_weight_adj,
            edge_index_sim=edge_index_sim,
            edge_weight_sim=edge_weight_sim,
            use_attention_fusion=True,
        )

    herb_index_df = data_bundle["herb_index"].copy()
    disease_index_df = data_bundle["disease_index"].copy()

    herb_name_col = pick_name_col(herb_index_df, ["herb_name", "name", "herbName", "common_name"], "herbid")
    disease_name_col = pick_name_col(
        disease_index_df,
        ["disease_name", "name", "diseaseName", "common_name"],
        "diseaseId",
    )

    herb_index_df["herb_name"] = herb_index_df[herb_name_col].astype(str)
    disease_index_df["disease_name"] = disease_index_df[disease_name_col].astype(str)

    herb_degree = R.sum(axis=1)
    top3_herb_idx = np.argsort(-herb_degree)[:3]

    top3_rows = []
    pred_rows = []

    for herb_rank, h_idx_val in enumerate(top3_herb_idx, start=1):
        herb_id = herb_index_df.iloc[h_idx_val]["herbid"]
        herb_name = herb_index_df.iloc[h_idx_val]["herb_name"]
        degree_val = int(herb_degree[h_idx_val])

        top3_rows.append(
            {
                "herb_rank": herb_rank,
                "herb_idx": int(h_idx_val),
                "herbid": herb_id,
                "herb_name": herb_name,
                "positive_degree": degree_val,
            }
        )

        neg_d_idx = np.flatnonzero(R[h_idx_val] == 0)
        h_tensor = torch.full((len(neg_d_idx),), int(h_idx_val), dtype=torch.long, device=device)
        d_tensor = torch.tensor(neg_d_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            probs = torch.sigmoid(model.score_pairs(herb_z, dis_z, h_tensor, d_tensor)).detach().cpu().numpy()

        take = min(10, len(neg_d_idx))
        top_local = np.argpartition(probs, -take)[-take:]
        top_local = top_local[np.argsort(probs[top_local])[::-1]]

        for disease_rank, local_idx in enumerate(top_local, start=1):
            d_idx_val = int(neg_d_idx[local_idx])
            disease_id = disease_index_df.iloc[d_idx_val]["diseaseId"]
            disease_name = disease_index_df.iloc[d_idx_val]["disease_name"]
            score = float(probs[local_idx])

            pred_rows.append(
                {
                    "herb_rank": herb_rank,
                    "herb_idx": int(h_idx_val),
                    "herbid": herb_id,
                    "herb_name": herb_name,
                    "positive_degree": degree_val,
                    "disease_rank": disease_rank,
                    "disease_idx": d_idx_val,
                    "diseaseId": disease_id,
                    "disease_name": disease_name,
                    "predicted_probability": score,
                }
            )

    top3_df = pd.DataFrame(top3_rows)
    pred_df = pd.DataFrame(pred_rows)

    top3_path = output_dir / "top3_herbs_most_connected.csv"
    pred_path = output_dir / "top10_new_diseases_per_top3_herbs.csv"
    top3_df.to_csv(top3_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    print("saved:", top3_path)
    print("saved:", pred_path)
    print(top3_df.to_string(index=False))
    print(pred_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
