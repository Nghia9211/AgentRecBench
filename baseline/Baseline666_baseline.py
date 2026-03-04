import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
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
        # Prompt cấu trúc theo Figure 4 của Baseline666
        prompt = f"{task_description}"
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        return reasoning_result

class MyRecommendationAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)

    def workflow(self):
        # Xác định dataset hiện tại (task_set được truyền từ simulator)
        current_set = task_set.lower()
        
        # 1. Platform-aware feature selection (Theo mục A.4 trang 13)
        if current_set == "amazon":
            item_keys = ['item_id', 'name', 'stars', 'review_count', 'description']
            review_keys = ['item_id', 'rating', 'text', 'verified_purchase', 'timestamp']
        elif current_set == "yelp":
            item_keys = ['item_id', 'name', 'stars', 'review_count', 'attributes']
            review_keys = ['item_id', 'rating', 'text', 'useful', 'funny', 'cool']
        elif current_set == "goodreads":
            item_keys = ['item_id', 'title', 'author', 'publication_year', 'average_rating', 'description', 'similar_books']
            review_keys = ['item_id', 'rating', 'text', 'review_date', 'votes', 'comments', 'read_status']
        else:
            item_keys = ['item_id', 'name', 'stars', 'description']
            review_keys = ['item_id', 'rating', 'text']

        # 2. Thu thập thông tin User (Figure 4 yêu cầu user_id, review_count, friends, stars)
        user_data = self.interaction_tool.get_user(user_id=self.task['user_id'])

        if user_data:
            user_profile = {
                'user_id': user_data.get('user_id'),
                'review_count': user_data.get('review_count', 0),
                'friends_count': len(user_data.get('friends', [])) if user_data.get('friends') else 0,
                'average_stars': user_data.get('average_rating', 0)
            }
        else:
            # Fallback cho User Cold Start hoặc khi không tìm thấy user
            user_profile = {'user_id': self.task['user_id'], 'note': 'New user or no profile data'}


        # 3. Thu thập lịch sử Review (Review-side feature engineering)
        all_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
        if all_reviews is None:
            all_reviews = []
        candidate_ids = set(self.task['candidate_list'])
        
        history_reviews_list = []
        for r in all_reviews:
            if r.get('item_id') not in candidate_ids:
                filtered_r = {k: r.get(k) for k in review_keys if k in r}
                history_reviews_list.append(filtered_r)
        history_str = str(history_reviews_list)
        if num_tokens_from_string(history_str) > 8000:
            encoding = tiktoken.get_encoding("cl100k_base")
            history_str = encoding.decode(encoding.encode(history_str)[:8000])

        item_details = []
        for item_id in self.task['candidate_list']:
            item = self.interaction_tool.get_item(item_id=item_id)
            if item:
                filtered_item = {k: item.get(k) for k in item_keys if k in item}
                item_details.append(filtered_item)
            else:
                item_details.append({'item_id': item_id, 'note': 'No description available'})

        task_description = f"""
You are a real user on an online platform. Your profile information: {user_profile}
Your historical item review text and stars are as follows: {history_str}

Now you need to rank the following 20 items: {self.task['candidate_list']} 
'according to their match degree to your preference'.

Please rank the more interested items more front in your rank list.
The information of the above 20 candidate items is as follows: {item_details}

Your final output should be ONLY a ranked item list of {self.task['candidate_list']}
with the following format!
DO NOT introduce any other item ids! DO NOT output your analysis process!
The correct output format: [Sorted Candidate Item List]
        """.strip()

        result = self.reasoning(task_description)
        
        try:
            matches = re.findall(r"\[(.*?)\]", result, re.DOTALL)
            if matches:
                last_match = matches[-1]
                items = [i.strip().strip("'").strip('"') for i in last_match.split(',')]
                final_list = [i for i in items if i in candidate_ids]
                
                if len(final_list) < len(self.task['candidate_list']):
                    remaining = [i for i in self.task['candidate_list'] if i not in final_list]
                    final_list.extend(remaining)
                
                print(f"Successfully ranked {len(final_list)} items for {current_set}")
                return final_list[:20]
            
            return self.task['candidate_list'] # Fallback
        except Exception as e:
            print(f"Error parsing: {e}")
            return self.task['candidate_list']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebSocietySimulator with Baseline666 Agent")
    parser.add_argument('--task_set', type=str, default='amazon', choices=['amazon', 'yelp', 'goodreads'])
    parser.add_argument('--scenario', type=str, default='classic', choices=['classic', 'user_cold_start','item_cold_start'])
    
    args = parser.parse_args()
    task_set = args.task_set # Global variable for agent to access
    scenario = args.scenario
    
    # Load Simulator
    simulator = Simulator(data_dir="../dataset/output_data_all/", device="gpu", cache=True) 
    simulator.set_task_and_groundtruth(
        task_dir=f"../dataset/task/{scenario}/{task_set}/tasks", 
        groundtruth_dir=f"../dataset/task/{scenario}/{task_set}/groundtruth"
    )

    # Set Agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM
    load_dotenv()
    openai_api_key = os.getenv("OPEN_API_KEY")
    simulator.set_llm(OpenAILLM(api_key=openai_api_key))

    # Run
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)
    # agent_outputs = simulator.run_simulation(number_of_tasks=2, enable_threading=True, max_workers=1)

    # Evaluate
    evaluation_results = simulator.evaluate()
    with open(f'./results/{scenario}/evaluation_results_Baseline666_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"The evaluation_results for {task_set} is: {evaluation_results}")