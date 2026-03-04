import json
import re
import traceback
from typing import Optional

import torch
from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
import tiktoken
from .prompts import *
from .schemas import BlackboardMessage, ItemRankerContent, NLIContent, RankedItem, RecState
from .utils import (
    find_top_k_similar_items, normalize_item,
    get_gcn_latent_interests, get_user_understanding, get_user_summary,
)
from .metric import evaluate_hit_rate


class ARAGAgents:
    def __init__(self, model, score_model, rank_model, embedding_function, gcn_path):
        self.model            = model
        self.score_model      = score_model
        self.rank_model       = rank_model
        self.embedding_function = embedding_function
        self.gcn_embeddings   = None
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = 8000  # Keep under 16k context limit for safety

        if gcn_path:
            try:
                self.gcn_embeddings = torch.load(gcn_path)
                print("GCN embeddings loaded.")
            except Exception as e:
                print(f"WARNING: Could not load GCN embeddings: {e}")

    def _gt_path(self, state):
        return (
            f"C:/Users/Admin/Desktop/Document/AgenticCode/AgentRecBench"
            f"/dataset/task/user_cold_start/{state['task_set']}/groundtruth"
        )
    def _truncate_context_by_tokens(self, data_list, max_tokens=None):
        """
        Truncate a list of review/context items to fit within token limit.
        Returns a truncated list and the token count of the truncated content.
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens
            
        if not data_list:
            return [], 0
            
        # Convert list to string representation
        content_str = json.dumps(data_list, ensure_ascii=False)
        tokens = self.encoding.encode(content_str)
        
        if len(tokens) <= max_tokens:
            return data_list, len(tokens)
        
        # Truncate items one by one until we fit
        truncated_list = []
        current_tokens = 0
        
        for item in data_list:
            item_str = json.dumps(item, ensure_ascii=False)
            item_tokens = len(self.encoding.encode(item_str))
            
            if current_tokens + item_tokens > max_tokens:
                break
            
            truncated_list.append(item)
            current_tokens += item_tokens
        
        return truncated_list, current_tokens
    # ── 1. USER UNDERSTANDING ────────────────────────────────────────────────
    def user_understanding_agent(self, state: RecState):
        history_ids = re.findall(r"'item_id':\s*'([^']+)'", str(state['long_term_ctx']))

        long_term_ctx_trunc, lt_tokens = self._truncate_context_by_tokens(
            state['long_term_ctx'], 
            max_tokens=self.max_context_tokens - 500  
        )
        
        
        print(f"\n[Token Management] Long-term context: {lt_tokens} tokens (truncated from {len(long_term_ctx_trunc)} to {len(long_term_ctx_trunc)} items)")
        

        gcn_insight = get_gcn_latent_interests(
            history_ids, self.gcn_embeddings, state['candidate_list']
        )
        prompt     = create_uua_prompt(long_term_ctx_trunc, state['current_session'], gcn_insight)
        uua_output = self.model.invoke(prompt).content

        return {"blackboard": [BlackboardMessage(role="UserUnderStanding", content=uua_output)]}

    # ── 2. INITIAL RETRIEVAL ─────────────────────────────────────────────────
    def initial_retrieval(self, state: RecState):
        query    = f"User Preference: {get_user_understanding(state)}"
        top_k    = find_top_k_similar_items(
            query, state['candidate_list'], self.embedding_function, k=5
        )
        evaluate_hit_rate(
            index=state['idx'], stage="1_Initial_Retrieval",
            items=top_k, gt_folder=self._gt_path(state), task_set=state['task_set'],
        )
        return {'top_k_candidate': top_k}

    # ── 3. NLI AGENT ────────────────────────────────────────────────────────
    def nli_agent(self, state: RecState, config: Optional[RunnableConfig] = None):
        threshold  = (config or {}).get("configurable", {}).get("nli_threshold", 5.5)
        user_pref  = get_user_understanding(state)
        candidates = [normalize_item(i) for i in state['top_k_candidate']]

        if not candidates:
            return {'positive_list': [], "blackboard": []}

        prompts = [
            create_nli_prompt(item=c, user_preferences=user_pref, item_id=c['item_id'])
            for c in candidates
        ]
        outputs = self.score_model.batch(prompts)

        positive, messages = [], []
        print(f"\033[93m[NLI]\033[0m threshold={threshold}")
        for item, out in zip(candidates, outputs):
            passed = out.score >= threshold
            print(f"  {'✅' if passed else '❌'} {out.score:.1f} | {item.get('name','?')}")
            if passed:
                positive.append(item)
            messages.append(BlackboardMessage(role="NaturalLanguageInference",
                                              content=out, score=out.score))

        output_state = {
            'positive_list': positive, 
            "blackboard": messages
        }

        if not positive:
            output_state['final_rank_list'] = [str(item.get('item_id')) for item in state['candidate_list']]

        evaluate_hit_rate(
            index=state['idx'], stage="2_NLI_Filtering",
            items=positive, gt_folder=self._gt_path(state), task_set=state['task_set'],
        )
        return output_state

    # ── 4. CONTEXT SUMMARY ───────────────────────────────────────────────────
    def context_summary_agent(self, state: RecState):
        if not state.get('positive_list'):
            return {"blackboard": [BlackboardMessage(role="ContextSummary",
                                                     content="No positive items found.")]}

        positive_ids = {i['item_id'] for i in state['positive_list']}
        nli_msgs     = [m for m in state['blackboard'] if m.role == "NaturalLanguageInference"]

        scored_str = ""
        for msg in nli_msgs:
            if msg.content.item_id in positive_ids:
                item = next((i for i in state['positive_list']
                             if i['item_id'] == msg.content.item_id), None)
                if item:
                    scored_str += f"Item: {item}\nScore: {msg.score}/10\nRationale: {msg.content.rationale}\n---\n"

        prompt = create_context_summary_prompt(
            user_summary=get_user_understanding(state),
            items_with_scores_str=scored_str,
        )
        output = self.model.invoke(prompt).content
        return {'blackboard': [BlackboardMessage(role="ContextSummary", content=output)]}

    # ── 5. ITEM RANKER ───────────────────────────────────────────────────────
    def item_ranker_agent(self, state: RecState):
        to_rank   = state['positive_list']
        all_items = state['candidate_list']

        if not to_rank:
            return {'final_rank_list': [i.get('item_id') for i in all_items]}

        prompt = create_ranking_prompt(
            user_summary=get_user_understanding(state),
            context_summary=get_user_summary(state),
            items_to_rank=json.dumps(to_rank, indent=2, ensure_ascii=False),
        )

        result = None
        for attempt in range(2):
            try:
                result = self.rank_model.invoke(prompt)
                break                               
            except Exception as e:
                err_msg = str(e)
                print(f"⚠️  Ranker attempt {attempt+1} failed: {err_msg[:120]}")
                traceback.print_exc()
        if result:
            ranked = result.ranked_list
        else:
            ranked = [RankedItem(item_id=str(i.get('item_id')),
                                 name=str(i.get('name', 'Unknown')),
                                 description="Fallback") for i in to_rank]

        ranked_ids   = {str(r.item_id) for r in ranked}
        unranked_ids = [str(i['item_id']) for i in all_items if str(i['item_id']) not in ranked_ids]
        final_ids    = [str(r.item_id) for r in ranked] + unranked_ids

        print(f"🏆 Rank: {final_ids[:5]}... ({len(final_ids)} total)")
        return {
            'final_rank_list': final_ids,
            'blackboard': [BlackboardMessage(role="ItemRanker",
                                             content=result or "Fallback ranking used")],
        }

    # ── ROUTER ───────────────────────────────────────────────────────────────
    def should_proceed_to_summary(self, state: RecState):
        if not state.get('positive_list'):
            print("No positive items — stopping early.")
            return END
        return "continue"