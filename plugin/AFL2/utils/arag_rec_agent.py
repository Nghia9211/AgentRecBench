import os
import sys
import json
import time
import traceback
import tiktoken
from typing import List, Dict, Optional, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
afl2_dir    = os.path.dirname(current_dir)
plugin_dir  = os.path.dirname(afl2_dir)
root_dir    = os.path.dirname(plugin_dir)

sys.path.append(root_dir)
sys.path.append(afl2_dir)

_plugin_src = os.path.join(plugin_dir, 'src')
if _plugin_src not in sys.path:
    sys.path.insert(0, _plugin_src)

_baseline_dir = os.path.join(root_dir, 'baseline')
if _baseline_dir not in sys.path:
    sys.path.insert(0, _baseline_dir)

from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from ARAGgcnRetrie.agents import ARAGAgents
from ARAGgcnRetrie.graph_builder import GraphBuilder
from ARAGgcnRetrie.schemas import NLIContent, ItemRankerContent, RecState
from ARAGgcnRetrie.utils import normalize_item
from ARAGgcnRetrie.processing_input import ReviewProcessor

from utils.rw_process import write_jsonl, read_jsonl


def _num_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    try:
        return len(enc.encode(text))
    except Exception:
        return 0


def _truncate(text: str, max_tokens: int) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        return enc.decode(tokens[:max_tokens])
    return text


class ARAGRecAgent:
    # ------------------------------------------------------------------ #
    #  Shared class-level resources — load 1 lần cho toàn bộ process      #
    # ------------------------------------------------------------------ #
    _shared_embedding      = None
    _shared_vector_store   = None
    _shared_gcn_embeddings = None
    _shared_agents         = None
    _shared_workflow       = None

    # FIX: SASRec-related resources cũng được share ở class level
    _shared_sasrec_model   = None   # SASRec model instance
    _shared_id2name        = None   # inner_id → item name
    _shared_name2id        = None   # item name → inner_id
    _shared_id2rawid       = None   # inner_id → raw_id gốc
    _shared_seq_size       = None
    _shared_item_num       = None

    def __init__(self, args, item_name_map: Dict[str, str] = None):
        self.args          = args
        self.memory        = []
        self.info_list     = []
        self.item_name_map = item_name_map or {}

        api_key = args.api_key.strip('"') if args.api_key else ""
        self.llm = ChatOpenAI(
            model=args.model,
            temperature=args.temperature,
            openai_api_key=api_key,
        )

        # ── Embedding Model ──────────────────────────────────────────────
        if ARAGRecAgent._shared_embedding is None:
            print("[ARAGRecAgent] Loading Embedding Model...")
            embed_model = getattr(args, 'embed_model_name',
                                  "sentence-transformers/all-MiniLM-L6-v2")
            ARAGRecAgent._shared_embedding = HuggingFaceEmbeddings(
                model_name=embed_model)
        self.embedding_function = ARAGRecAgent._shared_embedding

        # ── FAISS ────────────────────────────────────────────────────────
        if ARAGRecAgent._shared_vector_store is None:
            faiss_path = getattr(args, 'faiss_db_path', None)
            if faiss_path and os.path.exists(faiss_path):
                print(f"[ARAGRecAgent] Loading FAISS DB from {faiss_path}...")
                ARAGRecAgent._shared_vector_store = FAISS.load_local(
                    folder_path=faiss_path,
                    embeddings=self.embedding_function,
                    allow_dangerous_deserialization=True,
                    distance_strategy="COSINE",
                )
        self.vector_store = ARAGRecAgent._shared_vector_store

        # ── GCN Embeddings ───────────────────────────────────────────────
        if ARAGRecAgent._shared_gcn_embeddings is None:
            import torch
            gcn_path = getattr(args, 'gcn_path', None)
            if gcn_path and os.path.exists(gcn_path):
                print(f"[ARAGRecAgent] Loading GCN Embeddings from {gcn_path}...")
                ARAGRecAgent._shared_gcn_embeddings = torch.load(
                    gcn_path, map_location='cpu')
        self.gcn_embeddings = ARAGRecAgent._shared_gcn_embeddings

        # ── ARAG Agents & Workflow ────────────────────────────────────────
        if ARAGRecAgent._shared_agents is None:
            print("[ARAGRecAgent] Initializing ARAG Agents & Workflow...")
            agents = ARAGAgents(
                model=self.llm,
                score_model=self.llm.with_structured_output(NLIContent),
                rank_model=self.llm.with_structured_output(ItemRankerContent),
                embedding_function=self.embedding_function,
                gcn_path=getattr(args, 'gcn_path', None),
            )
            builder = GraphBuilder(agent_provider=agents)
            ARAGRecAgent._shared_agents   = agents
            ARAGRecAgent._shared_workflow = builder.build()
        self.agents   = ARAGRecAgent._shared_agents
        self.workflow = ARAGRecAgent._shared_workflow

        # ── SASRec (FIX: load 1 lần, share qua class variable) ──────────
        if ARAGRecAgent._shared_sasrec_model is None:
            self._load_sasrec(args)
        # Expose các attribute để các method khác dùng giống UserModelAgent
        self.sasrec_model = ARAGRecAgent._shared_sasrec_model
        self.id2name      = ARAGRecAgent._shared_id2name
        self.name2id      = ARAGRecAgent._shared_name2id
        self.id2rawid     = ARAGRecAgent._shared_id2rawid
        self.seq_size     = ARAGRecAgent._shared_seq_size
        self.item_num     = ARAGRecAgent._shared_item_num

        self._load_build_memory_template()

    # ------------------------------------------------------------------ #
    #  SASRec loader — chỉ chạy 1 lần nhờ class-level guard              #
    # ------------------------------------------------------------------ #
    @classmethod
    def _load_sasrec(cls, args):
        import torch
        import pandas as pd
        from utils.model import SASRec

        print("[ARAGRecAgent] Loading SASRec model (first time only)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_statis = pd.read_pickle(
            os.path.join(args.data_dir, 'data_statis.df'))
        seq_size = data_statis['seq_size'][0]
        item_num = data_statis['item_num'][0]

        sasrec = SASRec(64, item_num, seq_size, 0.1, device)
        sasrec.to(device)

        checkpoint = torch.load(args.model_path, map_location=device,
                                weights_only=False)
        if isinstance(checkpoint, dict):
            state = checkpoint.get('model_state_dict', checkpoint)
        else:
            state = checkpoint
        sasrec.load_state_dict(state)
        sasrec.eval()
        print("[ARAGRecAgent] SASRec loaded successfully.")

        # id2name / name2id / id2rawid
        id2name, name2id, id2rawid = {}, {}, {}
        item_path = os.path.join(args.data_dir, 'id2name.txt')
        with open(item_path, 'r', encoding='utf-8') as f:
            for line in f:
                ll = line.strip('\n').split('::')
                id2name[int(ll[0])]    = ll[1].strip()
                name2id[ll[1].strip()] = int(ll[0])

        rawid_path = os.path.join(args.data_dir, 'id2rawid.txt')
        if os.path.exists(rawid_path):
            with open(rawid_path, 'r', encoding='utf-8') as f:
                for line in f:
                    ll = line.strip('\n').split('::')
                    if len(ll) >= 2:
                        id2rawid[int(ll[0])] = ll[1].strip()
        else:
            print("[ARAGRecAgent] WARNING: id2rawid.txt not found.")

        # Ghi vào class variables
        cls._shared_sasrec_model = sasrec
        cls._shared_id2name      = id2name
        cls._shared_name2id      = name2id
        cls._shared_id2rawid     = id2rawid
        cls._shared_seq_size     = seq_size
        cls._shared_item_num     = item_num
        cls._shared_device       = device

    # ------------------------------------------------------------------ #
    #  SASRec inference — dùng shared model, không tạo instance mới       #
    # ------------------------------------------------------------------ #
    def model_generate(self, seq, len_seq, candidates) -> str:
        """Collaborative hint cho RecAgent — giống UserModelAgent.model_generate."""
        import torch
        import numpy as np

        device = ARAGRecAgent._shared_device
        states = torch.LongTensor([seq]).to(device)
        prediction = self.sasrec_model.forward_eval(
            states, np.array([len_seq]))

        sampling_idx = [True] * self.item_num
        for i in candidates:
            sampling_idx[i] = False
        sampling_idxs = torch.stack(
            [torch.tensor(sampling_idx)], dim=0)
        prediction = (prediction.cpu().detach()
                      .masked_fill(sampling_idxs,
                                   prediction.min().item() - 1))
        _, topK = prediction.topk(len(candidates), dim=1,
                                  largest=True, sorted=True)
        name_list = [self.id2name[i] for i in topK.numpy()[0]]
        len_ret = max(1, int(len(name_list) / 4))
        return ', '.join(name_list[:len_ret])

    def _load_build_memory_template(self):
        if 'amazon' in self.args.data_dir:
            from constant.amazon_prior_model_prompt import rec_build_memory
        elif 'goodreads' in self.args.data_dir:
            from constant.goodreads_prior_model_prompt import rec_build_memory
        elif 'yelp' in self.args.data_dir:
            from constant.yelp_prior_model_prompt import rec_build_memory
        else:
            rec_build_memory = "Round {}: Recommended {} because {}. Rejected because {}."
        self.rec_build_memory = rec_build_memory


    def _build_candidate_dicts(self, data: dict) -> List[dict]:
            from ARAGgcnRetrie.utils import normalize_item   # dùng normalize_item có sẵn

            tool       = data.get('interaction_tool')
            cans_ids   = data.get('cans', [])
            cans_names = data.get('cans_name', [])
            id2rawid   = data.get('id2rawid', {})
            item_list  = []

            for i, inner_id in enumerate(cans_ids):
                fallback_name = cans_names[i] if i < len(cans_names) else str(inner_id)
                raw_id = id2rawid.get(int(inner_id))

                raw_item = None
                if tool and raw_id:
                    raw_item = tool.get_item(item_id=str(raw_id))
                    if 'goodreads' in self.args.data_dir:
                        keys_to_extract = ["item_id","title", "description", "authors", "series", "popular_shelves_topk", "average_rating", "ratings_count", "text_reviews_count", "similar_books"]
                        filtered_item = {key: raw_item[key] for key in keys_to_extract if key in raw_item}
                        item_list.append(filtered_item)
                    else :
                        item_list.append(raw_item)
                # if raw_item and isinstance(raw_item, dict):
                #     # normalize_item tự dò alias cho item_id và name
                #     norm = normalize_item(raw_item)
                #     # Nếu normalize vẫn không tìm được → dùng fallback
                #     if norm.get('item_id', 'unknown') == 'unknown':
                #         norm['item_id'] = raw_id or str(inner_id)
                #     if norm.get('name', 'Unknown') == 'Unknown':
                #         norm['name'] = fallback_name
                #     item_list.append(norm)
                # else:
                #     if not raw_id:
                #         print(f"[ARAGRecAgent] No raw_id mapping for inner_id={inner_id}")
                #     else:
                #         print(f"[ARAGRecAgent] No item found for raw_id={raw_id}")
                #     # Fallback thuần: tạo dict tối thiểu đã chuẩn hoá
                #     item_list.append(normalize_item({
                #         'item_id': raw_id or str(inner_id),
                #         'name':    fallback_name,
                #     }))

            # print(f"[ARAGRecAgent] Candidate list (len={len(item_list)}): "
                # f"{item_list}\n")
            return item_list

    def _get_filtered_reviews(self, data: dict) -> list:
        tool    = data.get('interaction_tool')
        user_id = str(data.get('id', ''))

        if tool is None:
            print(f"[ARAGRecAgent] WARNING: interaction_tool not found in data for user {user_id}")
            return []

        all_reviews   = tool.get_reviews(user_id=user_id)
        candidate_ids = set(str(c) for c in data.get('cans', []))

        filtered = [r for r in all_reviews if r.get('item_id') not in candidate_ids]
        return filtered

    def _build_long_term_context(self, data: dict) -> str:
        """Giống file 2:
            self.processor.load_reviews(filtered_reviews[-15:])
            self.processor.process_and_split()
            long_term_ctx = self.processor.long_term_context
        """
        user_id  = str(data.get('id', ''))
        task_set = self._detect_task_set()
        filtered_reviews = self._get_filtered_reviews(data)

        if not filtered_reviews:
            base_ctx = "This is a new/cold-start user with no prior interaction history."
        else:
            # Truncate history string nếu quá dài (giống file 2 truncate tới 12000 tokens)
            history_str = str(filtered_reviews)
            if _num_tokens(history_str) > 12000:
                history_str = _truncate(history_str, 12000)

            processor = ReviewProcessor(target_source=task_set)
            processor.load_reviews(filtered_reviews[-15:])
            processor.process_and_split()
            base_ctx = processor.long_term_context
            print(f"[ARAGRecAgent] long_term_ctx (user {user_id}): {base_ctx[:120]}...")

        # Gắn rejection memory từ các round trước

        base_ctx_str = "\n".join([str(i) for i in base_ctx]) 
        if self.memory:
            base_ctx_str += "\n\n--- Previous Recommendation Attempts (REJECTED) ---\n"
            base_ctx_str += "\n".join(self.memory)
        

        return base_ctx_str

    def _build_current_session(self, data: dict) -> str:
        """Giống file 2:
            self.processor.load_reviews(filtered_reviews[-15:])
            self.processor.process_and_split()
            current_session = self.processor.short_term_context
        """
        task_set = self._detect_task_set()
        prior    = data.get('prior_answer', '')
        is_cold  = data.get('len_seq', 0) < 3

        filtered_reviews = self._get_filtered_reviews(data)

        if not filtered_reviews or is_cold:
            return (
                "Cold-start scenario: No reliable session data. "
                f"Sequential model prior suggestion (low confidence): {prior or 'N/A'}. "
                "Rely primarily on item semantics and collaborative graph signals."
            )

        processor = ReviewProcessor(target_source=task_set)
        processor.load_reviews(filtered_reviews[-15:])
        processor.process_and_split()
        session_ctx = processor.short_term_context

        # Truncate nếu quá dài (giống file 2 truncate tới 8000 tokens)
        if _num_tokens(session_ctx) > 8000:
            session_ctx = _truncate(session_ctx, 8000)

        # if prior:
        #     session_ctx = prior + " Sequential model also suggests: {prior}."
        
        # print(f"Current Session {session_ctx} \n\n")

        return session_ctx

    # ------------------------------------------------------------------
    # Core act() — chạy ARAG pipeline
    # ------------------------------------------------------------------
    def act(self, data: dict, reason=None, item=None):
        """Run ARAG pipeline, return (explanation: str, ranked_names: list)."""
        try:
            candidate_dicts = self._build_candidate_dicts(data)
            long_term_ctx   = self._build_long_term_context(data)
            current_session = self._build_current_session(data)

            is_cold       = data.get('len_seq', 0) < 3
            nli_threshold = getattr(self.args, 'nli_threshold', 5.5)
            if is_cold:
                nli_threshold = min(nli_threshold, 3.5)

            initial_state = {
                "idx":             0,
                "task_set":        self._detect_task_set(),
                "user_id":         str(data.get('id', '')),
                "long_term_ctx":   long_term_ctx,
                "current_session": current_session,
                "blackboard":      [],
                "candidate_list":  candidate_dicts,
            }

            print(f"\n[ARAGRecAgent] Running pipeline for user {data.get('id')} "
                  f"(cold={is_cold}, nli_thresh={nli_threshold:.1f}, "
                  f"round={len(self.memory)+1})")

            run_config  = {"configurable": {"nli_threshold": nli_threshold}}
            final_state = self.workflow.invoke(initial_state, config=run_config)

            # ---- PATCH: safe id_to_name construction ----
            id_to_name = {}
            for cd in candidate_dicts:
                cid  = cd.get('item_id') or cd.get('id')
                name = cd.get('name') or cd.get('title') or str(cid)
                if cid is not None:
                    id_to_name[str(cid)] = name

            ranked_names = []
            for rid in final_state.get('final_rank_list', []):
                name = id_to_name.get(str(rid), str(rid))
                if name not in ranked_names:
                    ranked_names.append(name)
                if len(ranked_names) >= 5:
                    break

            # Pad to 5 if ARAG returns fewer
            if len(ranked_names) < 5:
                for cd in candidate_dicts:
                    name = cd.get('name') or cd.get('title') or cd.get('raw_title', '')
                    if name and name not in ranked_names:
                        ranked_names.append(name)
                    if len(ranked_names) >= 5:
                        break

            explanation = self._extract_explanation(final_state)
            print(f"[ARAGRecAgent] Done. Top items: {ranked_names[:5]}")
            return explanation, ranked_names[:5]

        except Exception as e:
            print(f"[ARAGRecAgent] ERROR in act(): {e}")
            traceback.print_exc()
            fallback_names  = data.get('cans_name', [])[:5]
            fallback_reason = "ARAG pipeline failed, using fallback order."
            return fallback_reason, fallback_names

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_task_set(self) -> str:
        for ds in ('amazon', 'yelp', 'goodreads'):
            if ds in self.args.data_dir:
                return ds
        return 'unknown'

    def _extract_explanation(self, final_state: dict) -> str:
        """Lấy explanation từ blackboard của ARAG."""
        bb = final_state.get('blackboard', [])
        for msg in reversed(bb):
            if msg.role == "ItemRanker":
                if hasattr(msg.content, 'explanation'):
                    return msg.content.explanation[:500]
                if isinstance(msg.content, str):
                    return msg.content[:500]
        for msg in reversed(bb):
            if msg.role == "ContextSummary" and isinstance(msg.content, str):
                return msg.content[:500]
        return "Ranked by ARAG multi-agent pipeline based on semantic and collaborative signals."

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def build_memory(self, info: dict) -> str:
        rec_item_str = (
            ', '.join(info['rec_item_list'])
            if isinstance(info.get('rec_item_list'), list)
            else info.get('rec_item', '')
        )
        return self.rec_build_memory.format(
            info['epoch'], rec_item_str, info['rec_reason'], info['user_reason']
        )

    def update_memory(self, info: dict):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path: str):
        write_jsonl(path, self.info_list)

    def load_memory(self, path: str):
        self.info_list = read_jsonl(path)
        self.memory    = [self.build_memory(info) for info in self.info_list]