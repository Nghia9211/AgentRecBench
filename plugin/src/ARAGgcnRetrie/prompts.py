# ── prompts.py ────────────────────────────────────────────────────────────────
import yaml
from pathlib import Path


class PromptManager:
    def __init__(self):
        yaml_path = Path(__file__).parent / "prompts.yaml"
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self._prompts = yaml.safe_load(f)

    def get(self, key: str) -> str:
        return self._prompts.get(key, {}).get('template', "")


_pm = PromptManager()


def create_uua_prompt(long_term_ctx, current_session, gcn_insight: str) -> str:
    return _pm.get('user_understanding').format(
        long_term_context=long_term_ctx,
        current_session=current_session,
        gcn_behavior_insight=gcn_insight,
    )

def create_nli_prompt(item, user_preferences: str, item_id) -> str:
    return _pm.get('nli_scoring').format(
        item=item,
        user_preferences=user_preferences,
        item_id=item_id,
    )

def create_context_summary_prompt(user_summary: str, items_with_scores_str: str) -> str:
    return _pm.get('context_summary').format(
        user_summary=user_summary,
        items_with_scores_str=items_with_scores_str,
    )

def create_ranking_prompt(user_summary: str, context_summary: str, items_to_rank: str) -> str:
    return _pm.get('item_ranking').format(
        user_summary=user_summary,
        context_summary=context_summary,
        items_to_rank_str=items_to_rank,
    )
