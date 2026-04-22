from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ExperimentConfig:
    data_root: str = field(default_factory=lambda: str(Path(__file__).resolve().parents[2] / "data"))
    random_seed: int = 42
    n_folds: int = 5
    max_epochs: int = 500
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1e-4
    val_ratio_in_trainval: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 96
    out_dim: int = 64
    node_emb_dim: int = 96
    n_heads: int = 2
    topk_similarity: int = 20
    graph_pos_edge_cap: int = 300000
    eval_negative_ratio: float = 1.0
    max_positive_edges: Optional[int] = 300000
    contrastive_temperature: float = 0.2
    contrastive_weight: float = 0.1
    contrastive_max_pos: int = 1024
    dual_view_temperature: float = 0.2
    dual_view_weight: float = 0.15
    dual_view_max_nodes: int = 4096
    batch_size: int = 32768
    decision_threshold: float = 0.5
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
