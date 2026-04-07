"""
moe_fusion/
────────────
MoE Early Fusion Recommendation Pipeline.

Pipeline:
    C_R = C_seq ∪ C_gcn ∪ C_sem          (candidate retrieval)
    g(u,i) = softmax(MLP(x_{u,i}))       (gating network)
    s0(u,i) = g1*s_seq + g2*s_gcn + g3*s_sem  (early fusion)
    C_M = TopM(s0)                        (rough candidates)
    s_rerank = ARAG(C_M)                  (reranker)
    s1 = alpha*s0 + (1-alpha)*s_rerank    (final score)
    C_K = TopK(s1)                        (output)

Cách dùng nhanh — thay ARAGRecAgent trong dialogue_manager.py:

    from moe_fusion import MoERecAgent

    rec_agent = MoERecAgent(args)
    explanation, ranked_names = rec_agent.act(data)
    rec_agent.update_memory(info)

Train gating network offline:

    python -m moe_fusion.train_gating \\
        --data_dir ./data/amazon \\
        --model_path ./saved_models/amazon_best_model.pt \\
        --output_dir ./saved_models/moe
"""

from .moe_rec_agent      import MoERecAgent
from .config             import MoEConfig, RetrievalConfig, GatingConfig, ScoreConfig, get_config_for_dataset
from .seq_scorer         import SeqScorer
from .gcn_scorer         import GCNScorer
from .semantic_scorer    import SemanticScorer
from .candidate_retriever import CandidateRetriever
from .gating_network     import GatingNetwork, GatingMLP
from .moe_fusion         import MoEFusion
from .reranker           import Reranker
from .score_combiner     import ScoreCombiner

__all__ = [
    # Main entry point
    'MoERecAgent',
    # Config
    'MoEConfig', 'RetrievalConfig', 'GatingConfig', 'ScoreConfig',
    'get_config_for_dataset',
    # Scorers
    'SeqScorer', 'GCNScorer', 'SemanticScorer',
    # Pipeline stages
    'CandidateRetriever', 'GatingNetwork', 'GatingMLP',
    'MoEFusion', 'Reranker', 'ScoreCombiner',
]
