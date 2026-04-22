import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .config import ExperimentConfig


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_core_data(data_root: str) -> Dict[str, object]:
    root = Path(data_root)
    processed = root / "processed"
    raw = root / "raw" / "icam_net"
    summary = json.loads((processed / "dataset_summary.json").read_text(encoding="utf-8"))
    R = sparse.load_npz(processed / "R_hd.npz").astype(np.float32).toarray()
    herb_index = pd.read_csv(processed / "herb_index.csv")
    disease_index = pd.read_csv(processed / "disease_index.csv")
    hc = pd.read_csv(raw / "H_C_TCM.csv", usecols=["herbid", "cid"])
    dp = pd.read_csv(raw / "D_P_TCM.csv", usecols=["diseaseId", "geneId"])
    return {
        "summary": summary,
        "R": R,
        "herb_index": herb_index,
        "disease_index": disease_index,
        "H_C": hc,
        "D_P": dp,
    }


def build_binary_profile(
    edge_df: pd.DataFrame,
    left_col: str,
    right_col: str,
    left_vocab: np.ndarray,
    right_vocab: np.ndarray,
) -> sparse.csr_matrix:
    left_map = {v: i for i, v in enumerate(left_vocab)}
    right_map = {v: i for i, v in enumerate(right_vocab)}
    l = edge_df[left_col].map(left_map)
    r = edge_df[right_col].map(right_map)
    mask = l.notna() & r.notna()
    l_idx = l[mask].to_numpy(dtype=np.int64)
    r_idx = r[mask].to_numpy(dtype=np.int64)
    data = np.ones(len(l_idx), dtype=np.float32)
    return sparse.csr_matrix((data, (l_idx, r_idx)), shape=(len(left_vocab), len(right_vocab)), dtype=np.float32)


def knn_from_profile(profile: sparse.csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = profile.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32)
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(profile)
    dists, inds = nn.kneighbors(profile, return_distance=True)
    inds = inds[:, 1:]
    dists = dists[:, 1:]
    sims = 1.0 - dists
    sims = np.clip(sims, 0.0, 1.0).astype(np.float32)
    return inds.astype(np.int64), sims


def gip_knn(interaction: np.ndarray, axis: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x = interaction if axis == 0 else interaction.T
    x = x.astype(np.float32)
    n = x.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32)
    k_eff = max(1, min(k, n - 1))
    sq = np.sum(x * x, axis=1)
    gamma = 1.0 / np.mean(sq) if np.mean(sq) > 0 else 1.0
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean", algorithm="brute", n_jobs=-1)
    nn.fit(x)
    dists, inds = nn.kneighbors(x, return_distance=True)
    inds = inds[:, 1:]
    d2 = dists[:, 1:] ** 2
    sims = np.exp(-gamma * d2).astype(np.float32)
    return inds.astype(np.int64), sims


def combine_knn_graph(
    base_idx: np.ndarray,
    base_w: np.ndarray,
    extra_idx: Optional[np.ndarray],
    extra_w: Optional[np.ndarray],
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if extra_idx is None or extra_w is None:
        return base_idx, base_w
    n, k = base_idx.shape
    out_idx = np.zeros((n, k), dtype=np.int64)
    out_w = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        m = {}
        for j in range(k):
            nb = int(base_idx[i, j])
            m[nb] = m.get(nb, 0.0) + alpha * float(base_w[i, j])
        for j in range(k):
            nb = int(extra_idx[i, j])
            m[nb] = m.get(nb, 0.0) + (1.0 - alpha) * float(extra_w[i, j])
        items = sorted(m.items(), key=lambda t: t[1], reverse=True)[:k]
        out_idx[i, : len(items)] = [t[0] for t in items]
        out_w[i, : len(items)] = [t[1] for t in items]
    return out_idx, out_w


def build_adjacency_edge_index(
    R: np.ndarray,
    graph_pos_edge_cap: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_herb, n_dis = R.shape
    rows = []
    cols = []
    vals = []
    rr, cc = np.nonzero(R > 0)
    if len(rr) > graph_pos_edge_cap:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(rr), size=graph_pos_edge_cap, replace=False)
        rr = rr[keep]
        cc = cc[keep]

    for a, b in zip(rr, cc):
        u = int(a)
        v = n_herb + int(b)
        rows.append(u)
        cols.append(v)
        vals.append(1.0)
        rows.append(v)
        cols.append(u)
        vals.append(1.0)

    if len(rows) == 0:
        n_nodes = n_herb + n_dis
        rows = list(range(n_nodes))
        cols = list(range(n_nodes))
        vals = [1.0] * n_nodes

    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(np.asarray(vals, dtype=np.float32), dtype=torch.float32)
    return edge_index, edge_weight


def build_similarity_edge_index(
    herb_idx: np.ndarray,
    herb_w: np.ndarray,
    dis_idx: np.ndarray,
    dis_w: np.ndarray,
    n_herb: int,
    n_dis: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = []
    cols = []
    vals = []

    if herb_idx.size > 0:
        k_h = herb_idx.shape[1]
        for i in range(n_herb):
            for t in range(k_h):
                j = int(herb_idx[i, t])
                w = float(herb_w[i, t])
                if w > 0:
                    rows.append(i)
                    cols.append(j)
                    vals.append(w)

    if dis_idx.size > 0:
        k_d = dis_idx.shape[1]
        for i in range(n_dis):
            for t in range(k_d):
                j = int(dis_idx[i, t])
                w = float(dis_w[i, t])
                if w > 0:
                    u = n_herb + i
                    v = n_herb + j
                    rows.append(u)
                    cols.append(v)
                    vals.append(w)

    if len(rows) == 0:
        n_nodes = n_herb + n_dis
        rows = list(range(n_nodes))
        cols = list(range(n_nodes))
        vals = [1.0] * n_nodes

    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(np.asarray(vals, dtype=np.float32), dtype=torch.float32)
    return edge_index, edge_weight


def build_edge_index(
    R: np.ndarray,
    herb_idx: np.ndarray,
    herb_w: np.ndarray,
    dis_idx: np.ndarray,
    dis_w: np.ndarray,
    graph_pos_edge_cap: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    adj_index, adj_weight = build_adjacency_edge_index(R, graph_pos_edge_cap=graph_pos_edge_cap, seed=seed)
    sim_index, sim_weight = build_similarity_edge_index(herb_idx, herb_w, dis_idx, dis_w, n_herb=R.shape[0], n_dis=R.shape[1])
    edge_index = torch.cat([adj_index, sim_index], dim=1)
    edge_weight = torch.cat([adj_weight, sim_weight], dim=0)
    return edge_index, edge_weight


def prepare_knn_graphs(data_bundle: Dict[str, object], R_train: np.ndarray, k: int, use_extra_similarity: bool):
    herb_ids = data_bundle["herb_index"]["herbid"].to_numpy()
    dis_ids = data_bundle["disease_index"]["diseaseId"].to_numpy()
    comp_vocab = np.array(sorted(data_bundle["H_C"]["cid"].unique()))
    prot_vocab = np.array(sorted(data_bundle["D_P"]["geneId"].unique()))

    hc_profile = build_binary_profile(data_bundle["H_C"], "herbid", "cid", herb_ids, comp_vocab)
    dp_profile = build_binary_profile(data_bundle["D_P"], "diseaseId", "geneId", dis_ids, prot_vocab)

    h_idx_b, h_w_b = knn_from_profile(hc_profile, k)
    d_idx_b, d_w_b = knn_from_profile(dp_profile, k)

    if use_extra_similarity:
        h_idx_e, h_w_e = gip_knn(R_train, axis=0, k=k)
        d_idx_e, d_w_e = gip_knn(R_train, axis=1, k=k)
    else:
        h_idx_e, h_w_e = None, None
        d_idx_e, d_w_e = None, None

    h_idx, h_w = combine_knn_graph(h_idx_b, h_w_b, h_idx_e, h_w_e, alpha=0.5)
    d_idx, d_w = combine_knn_graph(d_idx_b, d_w_b, d_idx_e, d_w_e, alpha=0.5)
    return h_idx, h_w, d_idx, d_w


def collect_eval_edges(R: np.ndarray, cfg: ExperimentConfig):
    flat = R.reshape(-1)
    pos_idx = np.flatnonzero(flat == 1)
    neg_idx = np.flatnonzero(flat == 0)
    rng = np.random.default_rng(cfg.random_seed)

    if cfg.max_positive_edges is not None and len(pos_idx) > cfg.max_positive_edges:
        pos_idx = rng.choice(pos_idx, size=cfg.max_positive_edges, replace=False)

    neg_need = int(len(pos_idx) * cfg.eval_negative_ratio)
    neg_need = min(neg_need, len(neg_idx))
    neg_idx = rng.choice(neg_idx, size=neg_need, replace=False)

    eval_edges = np.concatenate([pos_idx, neg_idx])
    eval_labels = np.concatenate([
        np.ones(len(pos_idx), dtype=np.int32),
        np.zeros(len(neg_idx), dtype=np.int32),
    ])
    return eval_edges, eval_labels
