import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from utils.llm_provider import add_llm_args, build_llm_from_args
import re
import logging
import argparse
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except:
        return 0


class RecReasoning(ReasoningBase):
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        messages = [{"role": "user", "content": task_description}]
        return self.llm(messages=messages, temperature=0.1, max_tokens=1000)


class MyRecommendationAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)

    def workflow(self):
        current_set = task_set.lower()

        if current_set == "amazon":
            item_keys = ['item_id', 'name', 'stars', 'review_count', 'description']
            rev_keys  = ['item_id', 'rating', 'text', 'verified_purchase', 'timestamp']
        elif current_set == "yelp":
            item_keys = ['item_id', 'name', 'stars', 'review_count']
            rev_keys  = ['item_id', 'rating', 'text', 'useful', 'funny', 'cool']
        elif current_set == "goodreads":
            item_keys = ['item_id', 'title', 'average_rating', 'review_count']
            rev_keys  = ['item_id', 'rating', 'text', 'review_date', 'votes', 'comments', 'read_status']
        else:
            item_keys = ['item_id', 'name', 'stars']
            rev_keys  = ['item_id', 'rating', 'text']

        all_reviews   = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
        candidate_ids = set(self.task['candidate_list'])

        filtered_reviews = [
            {k: r.get(k) for k in rev_keys if k in r}
            for r in all_reviews if r.get('item_id') not in candidate_ids
        ]
        history_review = str(filtered_reviews)
        if num_tokens_from_string(history_review) > 8000:
            enc = tiktoken.get_encoding("cl100k_base")
            history_review = enc.decode(enc.encode(history_review)[:8000])

        item_list = []
        for item_id in self.task['candidate_list']:
            item = self.interaction_tool.get_item(item_id=item_id)
            if item:
                item_list.append({k: item.get(k) for k in item_keys if k in item})

        task_description = f"""
You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}.
Now you need to rank the following {len(self.task['candidate_list'])} items: {self.task['candidate_list']} according to their match degree to your preference.

Please rank the more interested items more front in your rank list.
The information of the above candidate items is as follows: {item_list}.

Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
DO NOT output your analysis process!
The correct output format: [Sorted Candidate Item List]
""".strip()

        result = self.reasoning(task_description)

        try:
            matches = re.findall(r"\[(.*?)\]", result, re.DOTALL)
            if matches:
                content = matches[-1].replace("'", "").replace('"', "")
                return [i.strip() for i in content.split(',') if i.strip() in candidate_ids][:20]
            return self.task['candidate_list']
        except:
            return self.task['candidate_list']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebSocietySimulator with DummyAgent")
    parser.add_argument('--task_set', default='amazon', choices=['amazon', 'yelp', 'goodreads'])
    parser.add_argument('--scenario', default='classic', choices=['classic', 'user_cold_start', 'item_cold_start'])
    add_llm_args(parser)
    args = parser.parse_args()

    task_set = args.task_set
    scenario = args.scenario

    load_dotenv()
    llm = build_llm_from_args(args, mode="simulator")

    simulator = Simulator(data_dir="../dataset/output_data_all/", device="gpu", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"../dataset/task/{scenario}/{task_set}/tasks",
        groundtruth_dir=f"../dataset/task/{scenario}/{task_set}/groundtruth",
    )
    simulator.set_agent(MyRecommendationAgent)
    simulator.set_llm(llm)

    agent_outputs      = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)
    evaluation_results = simulator.evaluate()

    os.makedirs(f'./results/{scenario}', exist_ok=True)
    with open(f'./results/{scenario}/evaluation_results_DummyAgent_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"The evaluation_results for {task_set} is: {evaluation_results}")