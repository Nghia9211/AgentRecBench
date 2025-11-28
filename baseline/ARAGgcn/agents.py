import json
from typing import Optional

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
import ast
from .prompts import *
from .schemas import (BlackboardMessage, ItemRankerContent, NLIContent,
                          RankedItem, RecState)
from .utils import find_top_k_similar_items,normalize_item_data, _get_gcn_similar_items
import torch

class ARAGAgents:
    def __init__(self, model, score_model, rank_model, embedding_function, gcn_path, graph_data_path = None):
        self.model = model
        self.score_model = score_model
        self.rank_model = rank_model
        self.embedding_function = embedding_function

        self.adj_matrix = None
        self.node_mapping = None 
        
        if graph_data_path:
            try:
                print(f"Loading Graph Structure from {graph_data_path}...")
                graph_data = torch.load(graph_data_path)
                
                self.adj_matrix = graph_data['adj_norm']
            
                raw_mapping = graph_data['node_item_mapping']
                self.str_to_idx = {str(val): i for i, val in enumerate(raw_mapping)}
                self.idx_to_str = {i: str(val) for i, val in enumerate(raw_mapping)}
                
                print("Graph Structure loaded successfully.")
            except Exception as e:
                print(f"WARNING: Could not load Graph data: {e}")

        self.gcn_embeddings = None
        if gcn_path:
            try:
                self.gcn_embeddings = torch.load(gcn_path)
                print("GCN Embeddings loaded successfully.")
            except Exception as e:
                print(f"WARNING: Could not load GCN embeddings: {e}")

    def initial_retrieval(self, state: RecState):
            lt_ctx = state['long_term_ctx']
            cur_ses = state['current_session']
            candidate_list = state['candidate_list']
            
            normalized_candidates = []
            for item in candidate_list:
                if isinstance(item, str):
                    try: item = ast.literal_eval(item)
                    except: 
                        try: item = json.loads(item)
                        except: continue
                
                norm_item = normalize_item_data(item)
                normalized_candidates.append(norm_item)


            query = f'Long-term Context : {lt_ctx} \n Current Session {cur_ses } \n '
            semantic_top_k = find_top_k_similar_items(query, candidate_list, self.embedding_function)

            num_nodes = self.adj_matrix.shape[0]
            user_preference_vector = torch.zeros(num_nodes).to(self.adj_matrix.device)
            
            found_anchors = 0
            for item in semantic_top_k:
                iid = str(item['item_id'])
                if iid in self.str_to_idx:
                    idx = self.str_to_idx[iid]
                    user_preference_vector[idx] = 1.0 
                    found_anchors += 1
            
            gcn_candidates = []
            if found_anchors > 0 and self.adj_matrix is not None:
                user_preference_vector = user_preference_vector / found_anchors
                
                # Hop 1 :
                prop_vec = user_preference_vector.unsqueeze(1) 
                prop_vec = torch.sparse.mm(self.adj_matrix, prop_vec)
                
                # Hop 2 (Item -> User -> Item 
                prop_vec = torch.sparse.mm(self.adj_matrix, prop_vec)
                
                # Hop 3 (Optional)
                # prop_vec = torch.sparse.mm(self.adj_matrix, prop_vec)
                
                scores = prop_vec.squeeze()
                
                for item in semantic_top_k:
                    iid = str(item['item_id'])
                    if iid in self.str_to_idx:
                        scores[self.str_to_idx[iid]] = -1.0 

                top_indices = torch.topk(scores, k=10).indices.tolist()
                
                for idx in top_indices:
                    original_id = self.idx_to_str.get(idx)
                    found_item = next((x for x in candidate_list if str(x['item_id']) == original_id), None)
                    if found_item:
                        gcn_candidates.append(found_item)

            combined_map = {str(item['item_id']): item for item in semantic_top_k + gcn_candidates}
            final_candidates = list(combined_map.values())
            print(f"Final Candidates : {final_candidates}")
            
            
            return {'top_k_candidate' : final_candidates, 'candidate_list': normalized_candidates}
    
    

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

            prompt = create_item_ranking_prompt(user_behavior_summary=user_understanding,
                context_summary=context_summary,
                items_to_rank=items_to_rank_str) 
            
            try:
                result_from_model = self.rank_model.invoke(prompt)
            except:
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