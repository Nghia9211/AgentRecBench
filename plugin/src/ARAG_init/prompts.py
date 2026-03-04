import yaml
import os
from pathlib import Path

class PromptManager:
    def __init__(self):
        yaml_path = Path(__file__).parent / "prompts.yaml"
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self._prompts = yaml.safe_load(f)

    def get_template(self, key):
        return self._prompts.get(key, {}).get('template', "")

_manager = PromptManager()

def create_summary_user_behavior_prompt(lt_ctx: str, cur_ses: str) -> str:
    template = _manager.get_template('user_understanding')
    return template.format(long_term_context=lt_ctx, current_session=cur_ses)

def create_assess_nli_score_prompt(item, lt_ctx: str, cur_ses: str, item_id) -> str:
    template = _manager.get_template('nli_scoring_v1')
    return template.format(item=item, long_term_context=lt_ctx, current_session=cur_ses, item_id=item_id)

def create_assess_nli_score_prompt2(item, user_preferences: str, item_id) -> str:
    template = _manager.get_template('nli_scoring_v2')
    return template.format(item=item, user_preferences=user_preferences, item_id=item_id)

def create_context_summary_prompt(user_summary: str, items_with_scores_str: str) -> str:
    template = _manager.get_template('context_summary')
    return template.format(user_summary=user_summary, items_with_scores_str=items_with_scores_str)

def create_item_ranking_prompt(user_summary, context_summary, items_to_rank) -> str:
    template = _manager.get_template('item_ranking')
    return template.format(user_summary=user_summary, context_summary=context_summary, items_to_rank_str=items_to_rank)