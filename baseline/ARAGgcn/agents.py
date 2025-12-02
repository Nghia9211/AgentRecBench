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
from .utils import normalize_item_data, generate_graph_context_string, perform_rag_retrieval

class ARAGAgents:
    def __init__(self, model, score_model, rank_model, embedding_function, gcn_path):
        self.model = model
        self.score_model = score_model
        self.rank_model = rank_model
        self.embedding_function = embedding_function

        # GCN Embeddings dùng để tạo Context mạng lưới
        self.gcn_embeddings = None
        if gcn_path:
            try:
                self.gcn_embeddings = torch.load(gcn_path)
                print("GCN Embeddings loaded successfully.")
            except Exception as e:
                print(f"WARNING: Could not load GCN embeddings: {e}")

    def initial_retrieval(self, state: RecState):
        """
        NEW FLOW:
        1. Extract User Info.
        2. GCN Context: Generate textual context from Graph Embeddings.
        3. RAG Retrieval: Use (User Info + GCN Context) to retrieve items via Semantic Search.
        """
        print("🔍 Starting Retrieval (GCN Context -> RAG)...")
        lt_ctx = state['long_term_ctx']
        cur_ses = state['current_session']
        candidate_list = state['candidate_list']
        
        # 1. Normalize Candidates
        normalized_candidates = []
        for item in candidate_list:
            if isinstance(item, str):
                try: item = ast.literal_eval(item)
                except: 
                    try: item = json.loads(item)
                    except: continue
            normalized_candidates.append(normalize_item_data(item))

        # 2. Extract User History IDs
        user_history_ids = []
        try:
            found_ids = re.findall(r"'item_id':\s*'([^']+)'", str(lt_ctx))
            user_history_ids.extend(found_ids)
        except:
            pass
        
        # --- BƯỚC QUAN TRỌNG: GCN CONTEXT GENERATION ---
        # "Hỏi" đồ thị xem user này liên quan đến những dạng item nào
        print("Generating Graph Context...")
        gcn_context_str = generate_graph_context_string(
            user_history_ids=user_history_ids,
            gcn_embeddings=self.gcn_embeddings,
            candidate_list=normalized_candidates,
            top_k_neighbors=5
        )
        
        print(f"Graph Context: {gcn_context_str[:150]}...") # Log 1 phần để kiểm tra

        # --- BƯỚC CUỐI: RAG RETRIEVAL ---
        # Tạo câu Query mở rộng (Augmented Query)
        rag_query = (
            f"User History: {lt_ctx}\n"
            f"Current Goal: {cur_ses}\n"
            f"System Graph Insights: {gcn_context_str}" # Đưa thông tin GCN vào Prompt tìm kiếm
        )
        
        print("Executing RAG Search with Graph-Augmented Query...")
        final_candidates = perform_rag_retrieval(
            augmented_query=rag_query,
            candidate_list=normalized_candidates,
            embedding_function=self.embedding_function,
            top_k=7
        )

        print(f"✅ Retrieved {len(final_candidates)} items via GCN->RAG pipeline.")
        
        return {'top_k_candidate': final_candidates, 'candidate_list': normalized_candidates}
    def nli_agent(self, state: RecState, config: Optional[RunnableConfig] = None):
        top_k_candidate = state['top_k_candidate']
        lt_ctx = state['long_term_ctx']
        cur_ses = state['current_session']

        configurable = config.get("configurable", {}) if config else {}
        threshold = configurable.get("nli_threshold", 5.5)

        if not top_k_candidate:
            return {'positive_list': [], "blackboard": []}

        prompts_list = [
            create_assess_nli_score_prompt(
                item=item, lt_ctx = lt_ctx, cur_ses = cur_ses, item_id =item['item_id'])
            for item in top_k_candidate
        ]
        all_nli_outputs = self.score_model.batch(prompts_list)

        positive_item_list = []
        new_blackboard_messages = []
        for item, nli_output in zip(top_k_candidate, all_nli_outputs):
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

        prompt = create_summary_user_behavior_prompt(
            lt_ctx = lt_ctx, cur_ses = cur_ses)
        uua_output = self.model.invoke(prompt).content

        uua_blackboard_message = BlackboardMessage(
            role="UserUnderStanding",
            content=uua_output
        )

        return {"blackboard": [uua_blackboard_message]}

    def context_summary_agent(self, state: RecState):
        blackboard = state['blackboard']
        positive_item = state['positive_list']

        if not positive_item:
            print("No positive items to summarize. Skipping.")
            return {"blackboard": [BlackboardMessage(role="ContextSummary", content="No positive items were found to summarize.")]}

        user_summary_msg = next((msg for msg in reversed(list(blackboard)) if msg.role == "UserUnderStanding"), None)
        user_summary_text = user_summary_msg.content if user_summary_msg else " No user summary found."

        nli_messages = [msg for msg in blackboard if msg.role == "NaturalLanguageInference"]

        positive_item_ids = {item['item_id'] for item in positive_item}

        items_with_scores_str = ""
        for msg in nli_messages:
            if msg.content.item_id in positive_item_ids:
                item_data = next(
                    (item for item in positive_item if item['item_id'] == msg.content.item_id), None)
                if item_data:
                    items_with_scores_str += (
                        f"Item: {item_data}\n"
                        f"NLI Score: {msg.score}/10\n"
                        f"Rationale: {msg.content.rationale}\n---\n"
                    )

        prompt = create_context_summary_prompt(
            user_summary=user_summary_text, items_with_scores_str=items_with_scores_str)
        csa_output = self.model.invoke(prompt).content

        csa_blackboard_message = BlackboardMessage(
            role="ContextSummary",
            content=csa_output
        )

        return {'blackboard': [csa_blackboard_message]}

    def item_ranker_agent(self, state: RecState):
            print("Item Ranking")

            blackboard = state['blackboard']
            items_to_rank = state['positive_list']
            candidate_list = state['candidate_list']

            if not items_to_rank:
                print("No items in the positive list to rank. Returning original candidate list.")
                final_list = [RankedItem(**item) for item in candidate_list]
                return {'final_rank_list': final_list}

            context_summary_msg = next((msg for msg in reversed(blackboard) if msg.role == "ContextSummary"), None)
            user_understanding_msg = next((msg for msg in reversed(blackboard) if msg.role == "UserUnderStanding"), None)

            context_summary = context_summary_msg.content if context_summary_msg else "No context summary available."
            user_understanding = user_understanding_msg.content if user_understanding_msg else "No user understanding available."

            items_to_rank_str = "\n\n".join([json.dumps(item, indent=2) for item in items_to_rank])

             # --- ĐOẠN CODE SỬA LẠI (Cắt ngắn description) ---
            simplified_items = []
            for item in items_to_rank:
                # Xử lý description: chuyển list thành str (nếu có) và cắt ngắn còn 200 ký tự
                desc_raw = item.get('description', '')
                if isinstance(desc_raw, list):
                    desc_str = " ".join(map(str, desc_raw))
                else:
                    desc_str = str(desc_raw)
                
                # Cắt ngắn description để Model không output quá dài
                short_desc = desc_str[:200] + "..." if len(desc_str) > 200 else desc_str

                simplified_items.append({
                    "item_id": str(item.get("item_id")), # Ép kiểu string luôn cho an toàn
                    "name": item.get("name"),
                    "category": item.get("category", "General"),
                    "description": short_desc 
                })

            items_to_rank_str = json.dumps(simplified_items, indent=2)


            prompt = create_item_ranking_prompt(user_behavior_summary=user_understanding,
                context_summary=context_summary,
                items_to_rank=items_to_rank_str) 
            
            try:
                result_from_model = self.rank_model.invoke(prompt)
            except Exception as e:
                print(f"❌ ERROR DETAILS: {e}") # In ra nội dung lỗi
                traceback.print_exc()          # In ra dòng code gây lỗi
                result_from_model = None
            
            if not result_from_model:
                print("⚠️ Model failed. Using original order.")
                ranked_positive_items = [
                    RankedItem(
                        item_id=str(i.get('item_id')),
                        name=str(i.get('title') or i.get('name') or 'Unknown'),
                        category="General",
                        description=str(i.get('description') if not isinstance(i.get('description'), list) else " ".join(map(str, i['description'])))
                    ) for i in items_to_rank
                ]

                result_from_model = ItemRankerContent(ranked_list=ranked_positive_items, explanation="Fallback strategy")
            else:
                ranked_positive_items = result_from_model.ranked_list

            ranked_item_ids = {item.item_id for item in ranked_positive_items}

            unranked_items = []
            for item in candidate_list: 
                if str(item['item_id']) not in ranked_item_ids:
                    unranked_items.append(
                        RankedItem(
                            item_id=item['item_id'],
                            name=item['name'],
                            category=item['category'],
                            description=item['description'] 
                        )
                    )

            final_full_ranked_list = ranked_positive_items + unranked_items

            result =  [item.item_id for item in final_full_ranked_list]


            item_ranking_message = BlackboardMessage(
                role="ItemRanker",
                content=result_from_model 
            )

            return {'final_rank_list':result , 'blackboard': [item_ranking_message]}
    
    def should_proceed_to_summary(self, state: RecState):
        blackboard = state['blackboard']
        
        has_uua_msg = any(msg.role == "UserUnderStanding" for msg in blackboard)
        has_nli_msg = any(msg.role == "NaturalLanguageInference" for msg in blackboard)

        if has_uua_msg and has_nli_msg:
            if not state['positive_list']:
                print("Synchronization check: No positive items found. Halting execution.")
                return END
            print("Synchronization check: Both branches complete. Proceeding to summary.")
            return "continue"
        else:
            print("Synchronization check: One or both branches have not completed. This should not happen.")
            return END