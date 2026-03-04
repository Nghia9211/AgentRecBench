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
    # Biến đếm dùng chung cho tất cả các instance của Agent để theo dõi task_id
    task_counter = 0 

    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)
        # Gán ID hiện tại cho agent này và tăng biến đếm
        self.current_task_idx = MyRecommendationAgent.task_counter
        MyRecommendationAgent.task_counter += 1

    def workflow(self):
        # --- Giữ nguyên phần khai báo biến và plan cũ của bạn ---
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user_data = {}
        item_list = []
        history_reviews_list = []
        
        gt_folder = f"../dataset/task/classic/{task_set}/groundtruth"
        # gt_folder = f"../dataset/task/user_cold_start/{task_set}/groundtruth"

        # --- Logic cũ để lấy thông tin User, Item, Review ---
        for sub_task in plan:
            if 'user' in sub_task['description']:
                user_raw = self.interaction_tool.get_user(user_id=self.task['user_id'])
                user_data = user_raw
            
            elif 'item' in sub_task['description']:
                for item_id in self.task['candidate_list']:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'description']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                    item_list.append(filtered_item)
            
            elif 'review' in sub_task['description']:
                all_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                candidate_ids = set(self.task['candidate_list'])
                history_reviews_list = [r for r in all_reviews if r.get('item_id') not in candidate_ids]

        # ==========================================================
        # --- BẮT ĐẦU PHẦN PHÂN TÍCH GROUNDTRUTH ---
        # ==========================================================
        
        gt_item_details = {}
        num_history_reviews = len(history_reviews_list)
        try:
            # 1. Đọc file groundtruth_i.json
            gt_file_path = os.path.join(gt_folder, f"groundtruth_{self.current_task_idx}.json")
            
            if os.path.exists(gt_file_path):
                with open(gt_file_path, 'r', encoding='utf-8') as f:
                    gt_json = json.load(f)
                    gt_id = gt_json["ground truth"]
                
                # 2. Lấy thông tin chi tiết của Groundtruth Item từ tool
                gt_item_raw = self.interaction_tool.get_item(item_id=gt_id)
                # Lọc bớt thông tin cho gọn
                keys_to_extract = ['item_id', 'name', 'title', 'stars', 'description', 'attributes', 'categories']
                gt_item_details = {k: gt_item_raw[k] for k in keys_to_extract if k in gt_item_raw}
            else:
                print(f"Warning: Groundtruth file not found at {gt_file_path}")
        except Exception as e:
            print(f"Error reading groundtruth: {e}")

        # 3. Xuất file phân tích
        analysis_dir = f"./analysis_{task_set}"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)

        analysis_file = os.path.join(analysis_dir, f"user_analysis_{self.current_task_idx}.json")
        
        analysis_output = {
            "task_index": self.current_task_idx,
            "user_id": self.task['user_id'],
            "history_count": num_history_reviews,
            "user_metadata": user_data,
            "user_history": history_reviews_list,
            "ground_truth_item": gt_item_details,
            "candidate_list_provided": self.task['candidate_list']
        }

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_output, f, indent=4, ensure_ascii=False)
        
        print(f"\033[94m[Analysis] Saved data for task {self.current_task_idx} to {analysis_file}\033[0m")

        # ==========================================================
        # --- KẾT THÚC PHẦN PHÂN TÍCH ---
        # ==========================================================

        # Giữ nguyên phần return cũ của bạn (ở đây tôi giả định là Dummy return để code không lỗi)
        return self.task['candidate_list']


if __name__ == "__main__":
    " Choose Dataset " 
    # task_set = "yelp" 
    task_set = "amazon"
    # task_set = "goodreads"
    
    " Load Dataset and simulator "
    simulator = Simulator(data_dir="../dataset/output_data_all/", device="gpu", cache=True) 

    " Load scenarios - Classic "
    # simulator.set_task_and_groundtruth(task_dir=f"../dataset/task/classic/{task_set}/tasks", groundtruth_dir=f"../dataset/task/classic/{task_set}/groundtruth")
    " Load scenarios - User Cold Start "
    simulator.set_task_and_groundtruth(task_dir=f"../dataset/task/user_cold_start/{task_set}/tasks", groundtruth_dir=f"../dataset/task/user_cold_start/{task_set}/groundtruth")
    " Load scenarios - Item Cold Start "
    # simulator.set_task_and_groundtruth(task_dir=f"../dataset/task/item_cold_start/{task_set}/tasks", groundtruth_dir=f"../dataset/task/item_cold_start/{task_set}/groundtruth")

    " Set Agent"
    simulator.set_agent(MyRecommendationAgent)

    " Set LLM client - CHANGE API KEY "
    load_dotenv()

    " -- OPEN AI -- "
    # openai_api_key = os.getenv("OPEN_API_KEY")
    # simulator.set_llm(OpenAILLM(api_key=openai_api_key))

    " -- GROQ -- "
    groq_api_key = os.getenv("GROQ_API_KEY3") # Change API-KEY HERE
    simulator.set_llm(GroqLLM(api_key = groq_api_key ,model="meta-llama/llama-4-scout-17b-16e-instruct"))


    " Run evaluation "
    " Note : If you set the number of tasks = None, the simulator will run all tasks."

    " Option 1: No Threading "
    agent_outputs = simulator.run_simulation(number_of_tasks=100, enable_threading=False)

    " Option 2: Threading - Max_workers = Numbers of Threads"
    # agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers = 10)

    " Evaluate Result "
    evaluation_results = simulator.evaluate()
    with open(f'./results/evaluation_results_TESTAgent_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
