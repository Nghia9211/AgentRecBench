import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
import argparse
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOT
from utils.llm_provider import add_llm_args, build_llm_from_args
import re
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except:
        return 0


class MyRecommendationAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.reasoning = ReasoningCOT(profile_type_prompt='', memory=None, llm=self.llm)
        self.memory    = MemoryDILU(llm=self.llm)

    def workflow(self):
        plan = [
            {'description': 'First I need to find user information'},
            {'description': 'Next, I need to find item information'},
            {'description': 'Next, I need to find review information'},
        ]

        user, item_list, history_review, filtered_reviews = '', [], '', []
        for sub_task in plan:
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                if num_tokens_from_string(user) > 8000:
                    enc = tiktoken.get_encoding("cl100k_base")
                    user = enc.decode(enc.encode(user)[:8000])

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
                filtered_reviews = [r for r in all_reviews if r.get('item_id') not in candidate_ids]
                history_review = str(filtered_reviews)
                if num_tokens_from_string(history_review) > 8000:
                    enc = tiktoken.get_encoding("cl100k_base")
                    history_review = enc.decode(enc.encode(history_review)[:8000])

        retrieved_memory = ""
        if filtered_reviews:
            for his in filtered_reviews:
                self.memory.addMemory(str(his))
            retrieved_memory = self.memory.retriveMemory(f"History review of {self.task['user_id']}")

        task_description = f"""
You are a recommendation agent. Your task is to recommend items for a user based on their profile, historical reviews, and a list of candidate items.

--- CANDIDATE ITEMS ---
{self.task['candidate_list']}

--- ITEMS DESCRIPTION ---
{item_list}

--- PAST EXPERIENCE (Retrieved from Memory) ---
{retrieved_memory}

--- YOUR TASK ---
Based on all the information above, analyze the user's preferences and the attributes of the candidate items.
Your final output MUST BE list of strings, where each string is an item_id from the candidate list.
The list should be ranked from the most recommended to the least recommended item for this user.
Do not include any other text, explanations, or markdown formatting around the list.
The correct output format: [Sorted Candidate Item List]
""".strip()

        result = self.reasoning(task_description)
        print('Meta Output:', result)

        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            result = match.group() if match else ''
            final  = eval(result)
            print('Processed Output:', final)
            return final
        except:
            print('format error')
            return ['']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebSocietySimulator with CoT Memory Agent")
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
    with open(f'./results/{scenario}/evaluation_results_CoTMemoryAgent_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"The evaluation_results for {task_set} is: {evaluation_results}")