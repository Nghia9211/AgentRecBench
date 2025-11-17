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
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """

        if task_set == "goodreads":
            task_type = "Goodreads"
            task_item = "book"
        elif task_set == "yelp":
            task_type = "Yelp"
            task_item = "business"
        elif task_set == "amazon":
            task_type = "Amazon"
            task_item = "product"
        else:
            # Fallback for any other dataset
            task_type = "platform"
            task_item = "item"
        
        plan = [
         {'description': 'First I need to find user information'},
         {'description': 'Next, I need to find item information'},
         {'description': 'Next, I need to find review information'}
         ]

        user = ''
        item_list = []
        history_review = ''
        for sub_task in plan:
            
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(user)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    user = encoding.decode(encoding.encode(user)[:12000])
                # print(user)
            elif 'item' in sub_task['description']:
                for n_bus in range(len(self.task['candidate_list'])):
                    item = self.interaction_tool.get_item(item_id=self.task['candidate_list'][n_bus])

                    # FILTERED ITEMS #
                    keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                    filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                    item_list.append(filtered_item)
                # print(f"Item_list : {item_list}")
                # for i, item in enumerate(item_list) :
                #     print(f"ITEM {i} : {item}\n\n")
            elif 'review' in sub_task['description']:
                history_review = str(self.interaction_tool.get_reviews(user_id=self.task['user_id']))
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        # print(f"History Review : {history_review} \n\n")
        # print(f"Candidate List : {self.task['candidate_list']}")

        # Dummy Core Workflow
        task_description = f'''
        ’You are a real human user on {task_type}, a platform for crowd-sourced
        {task_item} reviews’. 
        Here is your {task_type} profile and review history:
        {history_review}. 
        Your historical {task_item} reviews show your preference as
        follows: [’user_id’, ’review_count’, ’friends’, ’stars’...].
        Now you need to rank the following 20 {task_item}: {self.task['candidate_list']}
        according to their match degree to your preference. The information of the
        above 20 candidate {task_item} is as follows: 
        [’item_id’, ’name’, ’stars’,
        ’review_count’, ’attributes’, ’title’, ’average_rating’, ’rating_number’,
        ’description’, ’ratings_count’...]. 
        Your final output should be ONLY a
        ranked {task_item} list of {self.task['candidate_list']} with the following format, DO
        NOT introduce any other {task_item} ids!
        ’Please rank the more interested {task_item} more front in your rank list.’
        You should think step by step before your final answer. 
        **IMPORTANT** DO NOT output your analysis process! 
        Follow the correct final answer output format strictly,
        remember to output {task_item} ids instead of {task_item} names:
        [Sorted Candidate Item List]

        '''
        result = self.reasoning(task_description)
        # print(result)

        # try:
        #     print('Meta Output:',result)
        #     match = re.search(r"\[.*\]", result, re.DOTALL)
        #     if match:
        #         result = match.group()
        #     else:
        #         print("No list found.")
        #     print('Processed Output:',eval(result))
        #     # time.sleep(4)
        #     return eval(result)
        # except:
        #     print('format error')
        #     return ['']
        
        """"
        Extract Last List Item
        """
        try:
            #Tìm TẤT CẢ các chuỗi có dạng [...] trong kết quả
            matches = re.findall(r"(\[.*?\])", result, re.DOTALL)
            
            if matches:
                # Lấy chuỗi danh sách cuối cùng tìm được
                last_match_str = matches[-1]
                
                content_match = re.search(r"\[(.*)\]", last_match_str, re.DOTALL)
                content = content_match.group(1)

                items = [item.strip().strip("'\"") for item in content.split(',')]
                processed_list = [item for item in items if item]

                print('Processed Output List (from the end of response):', processed_list)
                return processed_list
            
            print("No list-like pattern found in the LLM response.")
            return ['']
        except Exception as e:
            print(f'An unexpected error occurred during parsing: {e}')
            return ['']

if __name__ == "__main__":
    " Choose Dataset " 
    # task_set = "yelp" 
    task_set = "amazon"
    # task_set = "goodreads"
    
    
    " Load Dataset and simulator "
    simulator = Simulator(data_dir="../dataset/output_data_all/", device="gpu", cache=True) 


    " Load scenarios - Classic "
    simulator.set_task_and_groundtruth(task_dir=f"../dataset/task/classic/{task_set}/tasks", groundtruth_dir=f"../dataset/task/classic/{task_set}/groundtruth")
    " Load scenarios - User Cold Start "
    # simulator.set_task_and_groundtruth(task_dir=f"../dataset/task/user_cold_start/{task_set}/tasks", groundtruth_dir=f"../dataset/task/user_cold_start/{task_set}/groundtruth")
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
    " -Task Type- Argument define only for RecHacker Baseline"
    " Option 1: No Threading "
    # agent_outputs = simulator.run_simulation(number_of_tasks=1, enable_threading=False, task_type = task_set)

    " Option 2: Threading - Max_workers = Numbers of Threads"
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers = 10)

    " Evaluate Result "
    evaluation_results = simulator.evaluate()
    with open(f'./results/evaluation_results_RecHacker_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
