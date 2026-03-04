import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, GroqLLM , OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase,ReasoningCOT
import re
import logging
import time
import argparse
import os

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

Task: {task_description}
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}

end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(example=few_shot, task_description=task_description, task_type=task_type, feedback=feedback)
        return prompt

class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}
'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        return reasoning_result

class MyRecommendationAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)

    def workflow(self):
        current_set = task_set.lower()
        
        # --- [DUMMY AGENT LOGIC] Platform-tailored Metadata (Trang 13) ---
        if current_set == "amazon":
            item_keys = ['item_id', 'name', 'stars', 'review_count', 'description']
            rev_keys = ['item_id', 'rating', 'text', 'verified_purchase', 'timestamp'] # Thêm verified_purchase
        elif current_set == "yelp":
            item_keys = ['item_id', 'name', 'stars', 'review_count']
            rev_keys = ['item_id', 'rating', 'text', 'useful', 'funny', 'cool'] # Thêm tương tác review
        elif current_set == "goodreads":
            item_keys = ['item_id', 'title', 'average_rating', 'review_count']
            rev_keys = ['item_id', 'rating', 'text', 'review_date', 'votes', 'comments', 'read_status'] # Thêm metadata sách
        else:
            item_keys = ['item_id', 'name', 'stars']
            rev_keys = ['item_id', 'rating', 'text']

        # 1. Thu thập dữ liệu
        all_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
        candidate_ids = set(self.task['candidate_list'])
        
        # Lọc history với metadata nâng cao
        filtered_reviews = [
            {k: r.get(k) for k in rev_keys if k in r}
            for r in all_reviews if r.get('item_id') not in candidate_ids
        ]
        
        history_review = str(filtered_reviews)
        if num_tokens_from_string(history_review) > 8000:
            encoding = tiktoken.get_encoding("cl100k_base")
            history_review = encoding.decode(encoding.encode(history_review)[:8000])

        item_list = []
        for item_id in self.task['candidate_list']:
            item = self.interaction_tool.get_item(item_id=item_id)
            if item:
                item_list.append({k: item.get(k) for k in item_keys if k in item})

        # --- [PROMPT] Theo Figure 5: DummyAgent's core workflow ---
        task_description = f'''
You are a real user on an online platform. Your historical item review text and stars are as follows: {history_review}. 
Now you need to rank the following {len(self.task['candidate_list'])} items: {self.task['candidate_list']} according to their match degree to your preference.

Please rank the more interested items more front in your rank list.
The information of the above candidate items is as follows: {item_list}.

Your final output should be ONLY a ranked item list of {self.task['candidate_list']} with the following format, DO NOT introduce any other item ids!
DO NOT output your analysis process!
The correct output format: [Sorted Candidate Item List]
        '''.strip()

        result = self.reasoning(task_description)
        
        # Robust Parsing (Giống phần trước)
        try:
            matches = re.findall(r"\[(.*?)\]", result, re.DOTALL)
            if matches:
                content = matches[-1].replace("'", "").replace('"', "")
                return [i.strip() for i in content.split(',') if i.strip() in candidate_ids][:20]
            return self.task['candidate_list']
        except:
            return self.task['candidate_list']


if __name__ == "__main__":
    " Choose Dataset " 
    # 1. Cấu hình Argument Parser
    parser = argparse.ArgumentParser(description="Run WebSocietySimulator with DummyAgent")
    parser.add_argument(
        '--task_set', 
        type=str, 
        default='amazon', 
        choices=['amazon', 'yelp', 'goodreads'],
        help='Name of the dataset to use (amazon, yelp, goodreads)'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='classic',
        choices=['classic', 'user_cold_start','item_cold_start'],
        help='Type of scenario to run (classic, user_cold_start,item_cold_start )'
    )
    

    args = parser.parse_args()
    task_set = args.task_set
    scenario = args.scenario
    
    " Load Dataset and simulator "
    simulator = Simulator(data_dir="../dataset/output_data_all/", device="gpu", cache=True) 


    " Load scenarios"
    simulator.set_task_and_groundtruth(task_dir=f"../dataset/task/{scenario}/{task_set}/tasks", groundtruth_dir=f"../dataset/task/{scenario}/{task_set}/groundtruth")

    " Set Agent"
    simulator.set_agent(MyRecommendationAgent)

    " Set LLM client - CHANGE API KEY "
    load_dotenv()

    " -- OPEN AI -- "
    openai_api_key = os.getenv("OPENAI_API_KEY")
    simulator.set_llm(OpenAILLM(api_key=openai_api_key))

    " -- GROQ -- "
    # groq_api_key = os.getenv("GROQ_API_KEY3") # Change API-KEY HERE
    # simulator.set_llm(GroqLLM(api_key = groq_api_key ,model="meta-llama/llama-4-scout-17b-16e-instruct"))


    " Run evaluation "
    " Note : If you set the number of tasks = None, the simulator will run all tasks."

    " Option 1: No Threading "
    # agent_outputs = simulator.run_simulation(number_of_tasks=1, enable_threading=False)

    " Option 2: Threading - Max_workers = Numbers of Threads"
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers = 10)

    " Evaluate Result "
    evaluation_results = simulator.evaluate()
    with open(f'./results/{scenario}/evaluation_results_DummyAgent_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
