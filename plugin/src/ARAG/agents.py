import json
from typing import Optional

from langgraph.graph import END
from langchain_core.runnables import RunnableConfig
import ast
from .prompts import *
from .schemas import (BlackboardMessage,
                          RankedItem, RecState)
from .metric import evaluate_hit_rate
from .utils import *


class ARAGAgents:
    def __init__(self, model, score_model, rank_model, embedding_function):
        self.model = model
        self.score_model = score_model
        self.rank_model = rank_model
        self.embedding_function = embedding_function

    def _get_gt_path(self, state):
        # return f"./dataset/task/user_cold_start/{state['task_set']}/groundtruth"
        return f"C:/Users/Admin/Desktop/Document/AgenticCode/AgentRecBench/dataset/task/user_cold_start/{state['task_set']}/groundtruth"

    def user_understanding_agent(self, state: RecState):
        lt_ctx = state['long_term_ctx']
        cur_ses = state['current_session']

        prompt = create_summary_user_behavior_prompt(lt_ctx = lt_ctx, cur_ses = cur_ses)
        response = self.model.invoke(prompt)
        uua_output = response.content

        uua_blackboard_message = BlackboardMessage(
            role="UserUnderStanding",
            content=uua_output
        )

        return {"blackboard": [uua_blackboard_message]}

    def initial_retrieval(self, state: RecState):

            user_understanding_msg = get_user_understanding(state)
            
            candidate_list = state['candidate_list']

            query = f' User Preference : {user_understanding_msg} \n '
    
            top_k_list = find_top_k_similar_items(query, candidate_list, self.embedding_function)

            evaluate_hit_rate(
                index = state['idx'],
                stage = "1_Initial_Retrieval",
                items = top_k_list,
                gt_folder = self._get_gt_path(state),
                task_set = state['task_set']
            )
            return {'top_k_candidate' : top_k_list}

    def nli_agent(self, state: RecState, config: Optional[RunnableConfig] = None):
        threshold = config.get("configurable", {}).get("nli_threshold", 5.5) if config else 5.5
        top_k_candidate_raw = state['top_k_candidate']
        user_understanding_msg = get_user_understanding(state)

        if not top_k_candidate_raw:
            return {'positive_list': [], "blackboard": []}

        top_k_candidate = []
        for item in top_k_candidate_raw:
            top_k_candidate.append(normalize_item_data(item))

        prompts_list = [
            create_assess_nli_score_prompt2(
                item=item, user_preferences = user_understanding_msg , item_id = item['item_id'])
            for item in top_k_candidate
        ]
        all_nli_outputs = self.score_model.batch(prompts_list)

        positive_item_list = []
        messages = []

        print(f"\033[93m[NLI Scoring]\033[0m Threshold: {threshold}")
        for item, nli_output in zip(top_k_candidate, all_nli_outputs):
            item_name = item.get('name') or item.get('title') or "Unkown"

            status = "✅ PASS" if nli_output.score >= threshold else "❌ FAIL"
            print(f"  - {status} | Score: {nli_output.score:.1f} | Item: {item_name}")
            print(f"---")
            print(f"Item: {item_name} | Score: {nli_output.score}")
            print(f"Rationale: {nli_output.rationale}") 
            if nli_output.score >= threshold:
                positive_item_list.append(item)

            messages.append(
                BlackboardMessage(
                    role="NaturalLanguageInference",
                    content=nli_output,
                    score=nli_output.score
                )
            )
        
        output_state = {
            'positive_list': positive_item_list, 
            "blackboard": messages
        }

        if not positive_item_list:
            output_state['final_rank_list'] = [str(item.get('item_id')) for item in state['candidate_list']]

        evaluate_hit_rate(
            index=state['idx'], 
            stage="2_NLI_Filtering", 
            items=positive_item_list,
            gt_folder=self._get_gt_path(state),
            task_set = state['task_set']
        )

        return output_state


    def context_summary_agent(self, state: RecState):
        blackboard = state['blackboard']
        positive_item = state['positive_list']

        if not positive_item:
            print("No positive items to summarize. Skipping.")
            return {"blackboard": [BlackboardMessage(role="ContextSummary", content="No positive items were found to summarize.")]}

        user_understanding_msg = get_user_understanding(state)

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
            user_summary=user_understanding_msg, items_with_scores_str=items_with_scores_str)
        # csa_output = self.model.invoke(prompt).content
        response = self.model.invoke(prompt)
        csa_output = response.content

        csa_blackboard_message = BlackboardMessage(
            role="ContextSummary",
            content=csa_output
        )

        return {'blackboard': [csa_blackboard_message]}

    def item_ranker_agent(self, state: RecState):
            print("\n--- [DEBUG ITEM RANKER] ---")
            items_to_rank = state['positive_list']
            candidate_list = state['candidate_list']

            if not items_to_rank:
                print("⚠️ [DEBUG] No items in positive_list. Skipping LLM call.")
                final_list = [item.get('item_id') for item in candidate_list]
                return {'final_rank_list': final_list}

            context_summary = get_user_summary(state)
            user_understanding = get_user_understanding(state)

            print(f"✅ [DEBUG] Items to rank count: {len(items_to_rank)}")
            print(f"✅ [DEBUG] User Summary Length: {len(user_understanding)}")

            items_to_rank_str = json.dumps(items_to_rank, indent=2, ensure_ascii=False)

            prompt = create_item_ranking_prompt(
                user_summary=user_understanding,
                context_summary=context_summary,
                items_to_rank=items_to_rank_str
            )
            result_from_model = None

            try:
                print("🚀 [DEBUG] Sending request Model...")
                result_from_model = self.rank_model.invoke(prompt)
                print(f"✅ [DEBUG] Model Response Received. Explanation len: {len(result_from_model.explanation if result_from_model else '')}")
            except Exception as e:
                print(f"❌ [DEBUG] LỖI KHI GỌI MODEL RANKER: {str(e)}")
                import traceback
                traceback.print_exc()


            if not result_from_model:
                print("⚠️ [DEBUG] Model failed/Returned None. Using fallback order.")
                ranked_positive_items = [
                    RankedItem(
                        item_id=str(i.get('item_id')),
                        name=str(i.get('title') or i.get('name') or 'Unknown'),
                        description="Fallback"
                    ) for i in items_to_rank
                ]
            else:
                ranked_positive_items = result_from_model.ranked_list

            ranked_item_ids = {str(item.item_id) for item in ranked_positive_items}
            unranked_items_ids = [str(item['item_id']) for item in candidate_list if str(item['item_id']) not in ranked_item_ids]

            final_result_ids = [str(item.item_id) for item in ranked_positive_items] + unranked_items_ids
            
            print(f"🏆 [DEBUG] Final Rank Order: {final_result_ids[:5]}... (Total: {len(final_result_ids)})")

            item_ranking_message = BlackboardMessage(
                role="ItemRanker",
                content=result_from_model if result_from_model else "Fallback ranking used"
            )

            return {'final_rank_list': final_result_ids, 'blackboard': [item_ranking_message]}

    def should_proceed_to_summary(self, state: RecState):
        if not state.get('positive_list') or len(state['positive_list']) == 0:
            print("No positive items found after NLI. Stopping.")
            return END
        
        return "continue"