"""
moe_fusion/config.py
────────────────────
Hyperparameters of MoE pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RetrievalConfig:
    top_seq: int = 20
    top_gcn: int = 20
    top_sem: int = 20
    top_M:   int = 20
    top_K:   int = 5


@dataclass
class GatingConfig:
    """Config MLP gating network."""
    input_dim:    int   = 4
    hidden_dims:  list  = field(default_factory=lambda: [16, 8])
    dropout:      float = 0.1
    lr:           float = 1e-3
    epochs:       int   = 30
    batch_size:   int   = 256
    weight_decay: float = 1e-4

    default_weights: list = field(
        default_factory=lambda: [1/3, 1/3, 1/3]
    )


@dataclass
class ScoreConfig:
    """
    s1(u,i) = alpha * s0(u,i) + (1 - alpha) * s_rerank(u,i)
    """
    alpha: float = 0.5

    dataset_alpha: Dict[str, float] = field(default_factory=lambda: {
        'yelp':      0.5,   
        'amazon':    0.5,
        'goodreads': 0.5,  
    })
    use_adaptive_alpha: bool = False


@dataclass
class MoEConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    gating:    GatingConfig    = field(default_factory=GatingConfig)
    scoring:   ScoreConfig     = field(default_factory=ScoreConfig)

    gating_model_path: str = None

    use_seq:      bool = True
    use_gcn:      bool = True
    use_semantic: bool = True
    use_reranker: bool = True


DEFAULT_CONFIG = MoEConfig()


def get_config_for_dataset() -> MoEConfig:
    cfg = MoEConfig()

    cfg.retrieval.top_seq = 20
    cfg.retrieval.top_gcn = 20
    cfg.retrieval.top_sem = 20
    cfg.retrieval.top_M   = 20
    cfg.retrieval.top_K   = 5
    cfg.gating.default_weights = [1/3, 1/3, 1/3]

    return cfg