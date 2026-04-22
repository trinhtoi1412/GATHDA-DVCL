from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
except Exception as exc:
    raise ImportError("torch_geometric is required.") from exc

from .config import ExperimentConfig


class GATMDAScalable(nn.Module):
    def __init__(self, n_herb: int, n_dis: int, cfg: ExperimentConfig):
        super().__init__()
        self.n_herb = n_herb
        self.n_dis = n_dis
        self.n_nodes = n_herb + n_dis
        self.node_emb = nn.Embedding(self.n_nodes, cfg.node_emb_dim)
        self.gat1 = GATConv(cfg.node_emb_dim, cfg.hidden_dim, heads=cfg.n_heads, concat=False)
        self.gat2 = GATConv(cfg.hidden_dim, cfg.hidden_dim, heads=cfg.n_heads, concat=False)
        self.gat3 = GATConv(cfg.hidden_dim, cfg.out_dim, heads=cfg.n_heads, concat=False)
        self.proj_h = nn.Linear(cfg.out_dim, cfg.out_dim)
        self.proj_d = nn.Linear(cfg.out_dim, cfg.out_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(cfg.out_dim * 2, cfg.out_dim),
            nn.ELU(),
            nn.Linear(cfg.out_dim, 2),
        )

    def encode_single(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.node_emb.weight
        h = F.elu(self.gat1(x, edge_index, edge_weight))
        h = F.elu(self.gat2(h, edge_index, edge_weight))
        h = self.gat3(h, edge_index, edge_weight)
        return h

    def fuse_views(
        self,
        z_adj: torch.Tensor,
        z_sim: Optional[torch.Tensor],
        use_attention_fusion: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if z_sim is None:
            return z_adj, None
        if use_attention_fusion:
            gate = torch.softmax(self.fusion_gate(torch.cat([z_adj, z_sim], dim=1)), dim=1)
            z = gate[:, :1] * z_adj + gate[:, 1:] * z_sim
            return z, gate
        return 0.5 * (z_adj + z_sim), None

    def encode(
        self,
        edge_index_adj: torch.Tensor,
        edge_weight_adj: torch.Tensor,
        edge_index_sim: Optional[torch.Tensor] = None,
        edge_weight_sim: Optional[torch.Tensor] = None,
        use_attention_fusion: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        z_adj = self.encode_single(edge_index_adj, edge_weight_adj)
        z_sim = None
        if edge_index_sim is not None and edge_weight_sim is not None:
            z_sim = self.encode_single(edge_index_sim, edge_weight_sim)
        z_fused, gate = self.fuse_views(z_adj, z_sim, use_attention_fusion=use_attention_fusion)
        herb_z = self.proj_h(z_fused[: self.n_herb])
        dis_z = self.proj_d(z_fused[self.n_herb :])
        aux = {"z_adj": z_adj, "z_sim": z_sim, "fusion_gate": gate}
        return herb_z, dis_z, aux

    def score_pairs(self, herb_z: torch.Tensor, dis_z: torch.Tensor, herb_idx: torch.Tensor, dis_idx: torch.Tensor) -> torch.Tensor:
        a = herb_z[herb_idx]
        b = dis_z[dis_idx]
        return torch.sum(a * b, dim=1)


def score_edge_batchwise(
    model: GATMDAScalable,
    edge_index_adj: torch.Tensor,
    edge_weight_adj: torch.Tensor,
    edge_index_sim: Optional[torch.Tensor],
    edge_weight_sim: Optional[torch.Tensor],
    use_attention_fusion: bool,
    eval_edges: np.ndarray,
    n_dis: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        herb_z, dis_z, _ = model.encode(
            edge_index_adj=edge_index_adj,
            edge_weight_adj=edge_weight_adj,
            edge_index_sim=edge_index_sim,
            edge_weight_sim=edge_weight_sim,
            use_attention_fusion=use_attention_fusion,
        )
        for st in range(0, len(eval_edges), batch_size):
            ed = min(st + batch_size, len(eval_edges))
            e = eval_edges[st:ed]
            h_eval = torch.tensor((e // n_dis).astype(np.int64), dtype=torch.long, device=device)
            d_eval = torch.tensor((e % n_dis).astype(np.int64), dtype=torch.long, device=device)
            s = torch.sigmoid(model.score_pairs(herb_z, dis_z, h_eval, d_eval)).detach().cpu().numpy()
            scores.append(s)
    return np.concatenate(scores)


def contrastive_pair_loss(
    herb_z: torch.Tensor,
    dis_z: torch.Tensor,
    herb_idx: torch.Tensor,
    dis_idx: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    max_pos: int,
) -> torch.Tensor:
    pos = torch.nonzero(labels > 0.5, as_tuple=False).squeeze(1)
    n_pos = int(pos.numel())
    if n_pos < 2:
        return herb_z.new_zeros(())
    if n_pos > max_pos:
        take = torch.randperm(n_pos, device=pos.device)[:max_pos]
        pos = pos[take]

    h_pos = F.normalize(herb_z[herb_idx[pos]], dim=1)
    d_pos = F.normalize(dis_z[dis_idx[pos]], dim=1)
    logits = (h_pos @ d_pos.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_hd = F.cross_entropy(logits, targets)
    loss_dh = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_hd + loss_dh)


def dual_view_contrastive_loss(
    z_adj: Optional[torch.Tensor],
    z_sim: Optional[torch.Tensor],
    temperature: float,
    max_nodes: int,
) -> torch.Tensor:
    if z_adj is None or z_sim is None:
        return torch.zeros((), device=z_adj.device if z_adj is not None else "cpu")
    n = z_adj.size(0)
    if n < 2:
        return z_adj.new_zeros(())
    if n > max_nodes:
        sel = torch.randperm(n, device=z_adj.device)[:max_nodes]
        a = z_adj[sel]
        b = z_sim[sel]
    else:
        a = z_adj
        b = z_sim

    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = (a @ b.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_ab = F.cross_entropy(logits, targets)
    loss_ba = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_ab + loss_ba)
