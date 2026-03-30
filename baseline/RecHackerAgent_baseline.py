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
        ts = task_set.lower()
        task_type = {"goodreads": "Goodreads", "yelp": "Yelp", "amazon": "Amazon"}.get(ts, "Platform")
        task_item = {"goodreads": "book",      "yelp": "business", "amazon": "product"}.get(ts, "item")

        plan = [
            {'description': 'First I need to find user information'},
            {'description': 'Next, I need to find item information'},
            {'description': 'Next, I need to find review information'},
        ]

        user, item_list, history_review = '', [], ''
        for sub_task in plan:
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                if num_tokens_from_string(user) > 12000:
                    enc = tiktoken.get_encoding("cl100k_base")
                    user = enc.decode(enc.encode(user)[:12000])

            elif 'item' in sub_task['description']:
                keys = ['item_id', 'name', 'stars', 'review_count', 'attributes',
                        'title', 'average_rating', 'rating_number', 'description',
                        'ratings_count', 'title_without_series']
                for item_id in self.task['candidate_list']:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    if item:
                        item_list.append({k: item[k] for k in keys if k in item})
                    else:
                        print(f"Warning: No data found for item_id: {item_id}. Skipping.")

            elif 'review' in sub_task['description']:
                all_reviews    = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                candidate_ids  = set(self.task['candidate_list'])
                filtered       = [r for r in all_reviews if r.get('item_id') not in candidate_ids]
                history_review = str(filtered)
                if num_tokens_from_string(history_review) > 16000:
                    enc = tiktoken.get_encoding("cl100k_base")
                    history_review = enc.decode(enc.encode(history_review)[:16000])

        task_description = f"""
You are a real human user on {task_type}, a platform for crowd-sourced {task_item} reviews.
Here is your {task_type} profile and review history: {history_review}.
Your historical {task_item} reviews show your preference as follows: ['user_id', 'review_count', 'friends', 'stars'...].
Now you need to rank the following 20 {task_item}: {self.task['candidate_list']} according to their match degree to your preference.
The information of the above 20 candidate {task_item} is as follows: {item_list}.

Your final output should be ONLY a ranked {task_item} list of {self.task['candidate_list']} with the following format.
DO NOT introduce any other {task_item} ids!
Please rank the more interested {task_item} more front in your rank list.
You should think step by step before your final answer.
IMPORTANT: DO NOT output your analysis process!
Remember to output {task_item} ids instead of {task_item} names.
The correct output format: [Sorted Candidate Item List]
""".strip()

        result = self.reasoning(task_description)

        try:
            matches = re.findall(r"(\[.*?\])", result, re.DOTALL)
            if matches:
                content = re.search(r"\[(.*)\]", matches[-1], re.DOTALL).group(1)
                items   = [i.strip().strip("'\"") for i in content.split(',')]
                print('Processed Output:', items)
                return [i for i in items if i]
            print("No list-like pattern found.")
            return ['']
        except Exception as e:
            print(f'Parsing error: {e}')
            return ['']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebSocietySimulator with RecHacker Agent")
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
        task_dir=f"../dataset/tasks2/{scenario}/{task_set}/tasks",
        groundtruth_dir=f"../dataset/tasks2/{scenario}/{task_set}/groundtruth",
    )
    simulator.set_agent(MyRecommendationAgent)
    simulator.set_llm(llm)

    agent_outputs      = simulator.run_simulation(number_of_tasks=100, enable_threading=True, max_workers=10)
    evaluation_results = simulator.evaluate()

    os.makedirs(f'./results/{scenario}', exist_ok=True)
    with open(f'./results/{scenario}/evaluation_results_RecHacker_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"The evaluation_results for {task_set} is: {evaluation_results}")