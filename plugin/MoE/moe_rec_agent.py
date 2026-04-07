"""
moe_fusion/moe_rec_agent.py
────────────────────────────
Feedback Loop 4.0: Soft Penalty (Phạt Mềm)
  - KHÔNG xóa bất kỳ item nào khỏi phễu.
  - Giảm điểm (Score Discounting) của những item bị reject ở round trước.
  - Bảo vệ Ground Truth khỏi sự "đánh giá sai" của User Simulator.
"""

import os
import sys
import threading
import traceback
from typing import Dict, List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config              import MoEConfig, get_config_for_dataset, DEFAULT_CONFIG
from seq_scorer          import SeqScorer
from gcn_scorer          import GCNScorer
from semantic_scorer     import SemanticScorer
from candidate_retriever import CandidateRetriever
from gating_network      import GatingNetwork
from moe_fusion          import MoEFusion
from reranker            import Reranker
from score_combiner      import ScoreCombiner

from utils.rw_process import write_jsonl, read_jsonl

class MoERecAgent:
    _shared_embedding      = None
    _shared_vector_store   = None
    _shared_gcn_embeddings = None
    _shared_sasrec_model   = None
    _shared_id2name:  Dict[int, str] = None
    _shared_name2id:  Dict[str, int] = None
    _shared_id2rawid: Dict[int, str] = None
    _shared_seq_size       = None
    _shared_item_num       = None
    _shared_device         = None
    _init_lock = threading.Lock()

    def __init__(self, args, moe_config=None, item_name_map=None, llm=None):
        self.args          = args
        self.memory:       List[str]  = []
        self.info_list:    List[dict] = []
        self.item_name_map = item_name_map or {}
        self.llm           = llm
        self.dataset = next((d for d in ['yelp', 'amazon', 'goodreads'] if d in getattr(args, 'data_dir', '')), 'amazon')
        self.cfg = moe_config or get_config_for_dataset(self.dataset)

        with MoERecAgent._init_lock:
            self._init_shared_resources(args)

        self.sasrec_model   = MoERecAgent._shared_sasrec_model
        self.gcn_embeddings = MoERecAgent._shared_gcn_embeddings
        self.embedding_fn   = MoERecAgent._shared_embedding
        self.vector_store   = MoERecAgent._shared_vector_store
        self.id2name        = MoERecAgent._shared_id2name
        self.name2id        = MoERecAgent._shared_name2id
        self.id2rawid       = MoERecAgent._shared_id2rawid
        self.item_num       = MoERecAgent._shared_item_num
        self.device         = MoERecAgent._shared_device

        self._build_pipeline()
        self._load_build_memory_template()

    @classmethod
    def _init_shared_resources(cls, args):
        import torch
        import pandas as pd
        from utils.model import SASRec

        if cls._shared_embedding is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            embed_model = getattr(args, 'embed_model_name', "sentence-transformers/all-MiniLM-L6-v2")
            cls._shared_embedding = HuggingFaceEmbeddings(model_name=embed_model)

        if cls._shared_vector_store is None:
            faiss_path = getattr(args, 'faiss_db_path', None)
            if faiss_path and os.path.exists(faiss_path):
                from langchain_community.vectorstores import FAISS
                cls._shared_vector_store = FAISS.load_local(folder_path=faiss_path, embeddings=cls._shared_embedding, allow_dangerous_deserialization=True, distance_strategy="COSINE")

        if cls._shared_gcn_embeddings is None:
            gcn_path = getattr(args, 'gcn_path', None)
            if gcn_path and os.path.exists(gcn_path):
                cls._shared_gcn_embeddings = torch.load(gcn_path, map_location='cpu', weights_only=True)

        if cls._shared_sasrec_model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data_statis = pd.read_pickle(os.path.join(args.data_dir, 'data_statis.df'))
            seq_size, item_num = data_statis['seq_size'][0], data_statis['item_num'][0]
            sasrec = SASRec(64, item_num, seq_size, 0.1, device).to(device)
            ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
            sasrec.load_state_dict(ckpt.get('model_state_dict', ckpt))
            sasrec.eval()
            id2name, name2id, id2rawid = {}, {}, {}
            with open(os.path.join(args.data_dir, 'id2name.txt'), encoding='utf-8') as f:
                for line in f:
                    ll = line.strip().split('::', 1)
                    if len(ll) == 2:
                        id2name[int(ll[0])], name2id[ll[1].strip()] = ll[1].strip(), int(ll[0])
            rawid_path = os.path.join(args.data_dir, 'id2rawid.txt')
            if os.path.exists(rawid_path):
                with open(rawid_path, encoding='utf-8') as f:
                    for line in f:
                        ll = line.strip().split('::')
                        if len(ll) >= 2: id2rawid[int(ll[0])] = ll[1].strip()
            cls._shared_sasrec_model, cls._shared_id2name, cls._shared_name2id, cls._shared_id2rawid = sasrec, id2name, name2id, id2rawid
            cls._shared_seq_size, cls._shared_item_num, cls._shared_device = seq_size, item_num, device

    def _build_pipeline(self):
        shared = {
            'model': self.sasrec_model, 'gcn_embeddings': self.gcn_embeddings,
            'vector_store': self.vector_store, 'embedding_function': self.embedding_fn,
            'id2name': self.id2name, 'name2id': self.name2id, 'id2rawid': self.id2rawid,
            'item_num': self.item_num, 'device': self.device,
        }
        self.seq_scorer = SeqScorer.from_shared(shared)
        self.gcn_scorer = GCNScorer.from_shared(shared) if self.gcn_embeddings is not None else None
        self.sem_scorer = SemanticScorer.from_shared(shared)
        self.retriever = CandidateRetriever(
            seq_scorer=self.seq_scorer, gcn_scorer=self.gcn_scorer, sem_scorer=self.sem_scorer,
            config=self.cfg.retrieval, use_seq=self.cfg.use_seq, use_gcn=self.cfg.use_gcn and self.gcn_scorer is not None, use_semantic=self.cfg.use_semantic and self.vector_store is not None,
        )
        self.gating = GatingNetwork(cfg=self.cfg.gating, model_path=getattr(self.args, 'gating_model_path', self.cfg.gating_model_path), device=self.device)
        self.fuser = MoEFusion(gating=self.gating, cfg=self.cfg)
        self.reranker = Reranker.from_shared(shared=shared, llm=self.llm, mode=getattr(self.args, 'reranker_mode', 'embed_only'), enabled=self.cfg.use_reranker, top_llm=getattr(self.args, 'reranker_top_llm', 15))
        self.combiner = ScoreCombiner(cfg=self.cfg.scoring)

    def _get_filtered_reviews(self, data: dict) -> list:
        tool = data.get('interaction_tool')
        if not tool: return []
        try:
            all_reviews = tool.get_reviews(user_id=str(data.get('id', '')))
            candidate_ids = set(str(c) for c in data.get('cans', []))
            return [r for r in all_reviews if str(r.get('item_id')) not in candidate_ids]
        except Exception: return []

    def act(self, data: dict, reason: str = None, item: str = None, epoch: int = None, rejected_items: List[str] = None) -> Tuple[str, List[str], dict]:
        user_id = data.get('id', 'unknown')
        if epoch is None: epoch = len(self.memory) + 1
        rejected_items = rejected_items or []

        try:
            gt_item = data.get('correct_answer', '').strip()
            if 'reviews' not in data:
                reviews = self._get_filtered_reviews(data)
                if reviews: data['reviews'] = reviews

            # 1. Retrieval (KHÔNG XÓA BẤT KỲ ITEM NÀO)
            union_names, signal_scores = self.retriever.retrieve_from_data(data)

            # 2. Fusion (KHÔNG XÓA)
            c_m, fused_scores, fusion_debug = self.fuser.fuse_from_data(data=data, signal_scores=signal_scores, debug=True)
            if not c_m: c_m = union_names[:self.cfg.retrieval.top_M]

            # 3. Reranker đọc Lời chê nhưng đánh giá trên toàn bộ tập c_m gốc
            ranked, rerank_scores, explanation = self.reranker.rerank(data=data, c_m=c_m, id2name=self.id2name, name2id=self.name2id, memory=self.memory)

            # 4. Combine Scores
            cm_fused = {name: fused_scores.get(name, 0.0) for name in c_m}
            c_k_raw, s1_scores, combine_debug = self.combiner.combine_from_pipeline(fused_scores=cm_fused, rerank_scores=rerank_scores, data=data, args=self.args, top_k=self.cfg.retrieval.top_M, epoch=epoch)

            # ── [FL 4.0] BƯỚC QUAN TRỌNG: SOFT PENALTY (PHẠT MỀM) ──
            PENALTY_FACTOR = 0.6 
            for item in rejected_items:
                if item in s1_scores:
                    s1_scores[item] = s1_scores[item] * PENALTY_FACTOR

            # 5. Sắp xếp lại danh sách sau khi đã phạt mềm
            c_k_sorted = sorted(c_k_raw, key=lambda x: s1_scores.get(x, 0.0), reverse=True)
            
            # Lấy Top K cuối cùng
            c_k = c_k_sorted[:self.cfg.retrieval.top_K]
            
            if len(c_k) < self.cfg.retrieval.top_K:
                for name in ranked:
                    if name not in c_k: c_k.append(name)
                    if len(c_k) >= self.cfg.retrieval.top_K: break

            debug_info = {
                'gt_item': gt_item, 'alpha': combine_debug.get('alpha', 0.0), 'top_M_size': len(c_m),
                'c_m_top_k_before_rerank': c_m[:self.cfg.retrieval.top_K], 'c_k_final_after_rerank': c_k,
                'scores_breakdown': {item: {'s0_moe': fused_scores.get(item, 0.0), 's_rerank': rerank_scores.get(item, 0.0) if rerank_scores else 0.0, 's1_final': s1_scores.get(item, 0.0)} for item in c_k}
            }

            return explanation or "MoE pipeline recommendation.", c_k, debug_info

        except Exception as e:
            traceback.print_exc()
            return "MoE pipeline failed.", data.get('cans_name', [])[:self.cfg.retrieval.top_K], {'error': str(e)}

    def build_memory(self, info: dict) -> str:
        rec_item_str = ', '.join(info['rec_item_list']) if isinstance(info.get('rec_item_list'), list) else info.get('rec_item', '')
        return self.rec_build_memory.format(info['epoch'], rec_item_str, info.get('rec_reason', ''), info.get('user_reason', ''))

    def update_memory(self, info: dict):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def _load_build_memory_template(self):
        try:
            if 'amazon' in self.args.data_dir: from constant.amazon_prior_model_prompt import rec_build_memory
            elif 'goodreads' in self.args.data_dir: from constant.goodreads_prior_model_prompt import rec_build_memory
            elif 'yelp' in self.args.data_dir: from constant.yelp_prior_model_prompt import rec_build_memory
            else: rec_build_memory = "Round {}: Recommended {} because {}. Rejected because {}."
        except ImportError:
            rec_build_memory = "Round {}: Recommended {} because {}. Rejected because {}."
        self.rec_build_memory = rec_build_memory

    def get_shared_sasrec(self) -> dict:
        return {'model': MoERecAgent._shared_sasrec_model, 'id2name': MoERecAgent._shared_id2name, 'name2id': MoERecAgent._shared_name2id, 'id2rawid': MoERecAgent._shared_id2rawid, 'seq_size': MoERecAgent._shared_seq_size, 'item_num': MoERecAgent._shared_item_num, 'device': MoERecAgent._shared_device}