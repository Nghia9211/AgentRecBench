import json
from typing import Optional
import traceback 
from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
import ast
import re
import torch
from .prompts import *
from .schemas import (BlackboardMessage, ItemRankerContent, NLIContent, RankedItem, RecState)
from .utils import normalize_item_data, generate_graph_context_string, perform_rag_retrieval, print_agent_step,get_gcn_latent_interests

class ARAGAgents:
    def __init__(self, model, score_model, rank_model, embedding_function, gcn_path):
        self.model = model
        self.score_model = score_model
        self.rank_model = rank_model
        self.embedding_function = embedding_function

        self.gcn_embeddings = None
        if gcn_path:
            try:
                self.gcn_embeddings = torch.load(gcn_path)
                print("GCN Embeddings loaded successfully.")
            except Exception as e:
                print(f"WARNING: Could not load GCN embeddings: {e}")

    def initial_retrieval(self, state: RecState):
        # Bước này bây giờ KHÔNG CẦN GCN nữa, vì GCN đã nằm trong 'UserUnderStanding'
        blackboard = state['blackboard']
        user_understanding_msg = next((msg for msg in reversed(blackboard) if msg.role == "UserUnderStanding"), None)
        
        # Query này đã được LLM 'nhúng' sẵn các gợi ý từ GCN một cách tự nhiên
        query = user_understanding_msg.content 

        # Semantic Search thuần túy
        top_k_list = perform_rag_retrieval(query, state['candidate_list'], self.embedding_function)

        return {'top_k_candidate': top_k_list}
    def nli_agent(self, state: RecState, config: Optional[RunnableConfig] = None):
        top_k_candidate = state['top_k_candidate']
        blackboard = state['blackboard']
        # Lấy User Preferences từ Blackboard
        user_understanding_msg = next((msg for msg in reversed(blackboard) if msg.role == "UserUnderStanding"), None)

        configurable = config.get("configurable", {}) if config else {}
        threshold = configurable.get("nli_threshold", 5.5)

        if not top_k_candidate:
            return {'positive_list': [], "blackboard": []}

        # Sử dụng create_assess_nli_score_prompt2 như bản cũ
        prompts_list = [
            create_assess_nli_score_prompt2(
                item=item, user_preferences=user_understanding_msg.content if user_understanding_msg else "", item_id=item['item_id'])
            for item in top_k_candidate
        ]
        all_nli_outputs = self.score_model.batch(prompts_list)

        positive_item_list = []
        new_blackboard_messages = []

        print(f"\033[93m[NLI Scoring]\033[0m Threshold: {threshold}")
        for item, nli_output in zip(top_k_candidate, all_nli_outputs):
            item_name = item.get('name') or item.get('title') or "Unknown"
            status = "✅ PASS" if nli_output.score >= threshold else "❌ FAIL"
            print(f"  - {status} | Score: {nli_output.score:.1f} | Item: {item_name}")
            
            if nli_output.score >= threshold:
                positive_item_list.append(item)

            new_blackboard_messages.append(
                BlackboardMessage(
                    role="NaturalLanguageInference",
                    content=nli_output,
                    score=nli_output.score
                )
            )
        return {'positive_list': positive_item_list, "blackboard": new_blackboard_messages}

    def user_understanding_agent(self, state: RecState):
            lt_ctx = state['long_term_ctx']
            cur_ses = state['current_session']
            candidate_list = state['candidate_list']

            # Lấy thông tin từ GCN trước
            user_history_ids = re.findall(r"'item_id':\s*'([^']+)'", str(lt_ctx))
            # GCN Insight không phải là danh sách để tìm kiếm, mà là mô tả về 'Gu'
            gcn_behavior_insight = get_gcn_latent_interests(user_history_ids, self.gcn_embeddings, candidate_list)

            # Prompt bây giờ nhận cả 2: Lịch sử văn bản + Phân tích đồ thị 
            prompt = create_summary_user_behavior_prompt2(lt_ctx, cur_ses, gcn_behavior_insight)
            
            uua_output = self.model.invoke(prompt).content
            
            print_agent_step("User Understanding", "Enhanced Profile with GCN Insights", uua_output)
            
            return {"blackboard": [BlackboardMessage(role="UserUnderStanding", content=uua_output)]}

    def context_summary_agent(self, state: RecState):
        blackboard = state['blackboard']
        positive_item = state['positive_list']

        if not positive_item:
            return {"blackboard": [BlackboardMessage(role="ContextSummary", content="No positive items found.")]}

        user_summary_msg = next((msg for msg in reversed(list(blackboard)) if msg.role == "UserUnderStanding"), None)
        user_summary_text = user_summary_msg.content if user_summary_msg else ""

        nli_messages = [msg for msg in blackboard if msg.role == "NaturalLanguageInference"]
        positive_item_ids = {item['item_id'] for item in positive_item}

        items_with_scores_str = ""
        for msg in nli_messages:
            if msg.content.item_id in positive_item_ids:
                item_data = next((item for item in positive_item if item['item_id'] == msg.content.item_id), None)
                if item_data:
                    items_with_scores_str += (f"Item: {item_data}\nNLI Score: {msg.score}/10\nRationale: {msg.content.rationale}\n---\n")

        prompt = create_context_summary_prompt(user_summary=user_summary_text, items_with_scores_str=items_with_scores_str)
        csa_output = self.model.invoke(prompt).content

        print_agent_step("Context Summary", "Context Summary Agent complete", csa_output)
        
        return {'blackboard': [BlackboardMessage(role="ContextSummary", content=csa_output)]}

    def item_ranker_agent(self, state: RecState):
            blackboard = state['blackboard']
            items_to_rank = state['positive_list']
            candidate_list = state['candidate_list']

            if not items_to_rank:
                return {'final_rank_list': [item['item_id'] for item in candidate_list]}

            context_summary_msg = next((msg for msg in reversed(blackboard) if msg.role == "ContextSummary"), None)
            user_understanding_msg = next((msg for msg in reversed(blackboard) if msg.role == "UserUnderStanding"), None)

            context_summary = context_summary_msg.content if context_summary_msg else ""
            user_understanding = user_understanding_msg.content if user_understanding_msg else ""

            simplified_items = []
            for item in items_to_rank:
                desc_raw = item.get('description', '')
                desc_str = " ".join(map(str, desc_raw)) if isinstance(desc_raw, list) else str(desc_raw)
                simplified_items.append({
                    "item_id": str(item.get("item_id")),
                    "name": item.get("name"),
                    "category": item.get("category", "General"),
                    "description": desc_str[:200] + "..."
                })

            prompt = create_item_ranking_prompt(user_summary=user_understanding, context_summary=context_summary, items_to_rank=json.dumps(simplified_items, indent=2)) 
            
            try:
                result_from_model = self.rank_model.invoke(prompt)
                print(f"✅ Ranking Process Done. Explanation: {result_from_model.explanation[:100]}...")
            except:
                result_from_model = None
            
            if not result_from_model:
                ranked_item_ids = [str(i['item_id']) for i in items_to_rank]
            else:
                ranked_item_ids = [str(item.item_id) for item in result_from_model.ranked_list]

            # Merge with unranked items to maintain full list
            all_ids = ranked_item_ids + [str(item['item_id']) for item in candidate_list if str(item['item_id']) not in ranked_item_ids]

            print(f"🏆 Final Rank Order: {all_ids[:5]}...")
            return {'final_rank_list': all_ids, 'blackboard': [BlackboardMessage(role="ItemRanker", content=result_from_model or "Fallback")]}
    
    def should_proceed_to_summary(self, state: RecState):
        if not state.get('positive_list') or len(state['positive_list']) == 0:
            print("No positive items found after NLI. Stopping.")
            return END
        return "continue"