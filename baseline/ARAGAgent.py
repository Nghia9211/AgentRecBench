import sys
import os

# Add Plugin Folder to import ARAG
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent 
from websocietysimulator.llm import LLMBase, InfinigenceLLM, GroqLLM , OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase,ReasoningCOT

import tiktoken
import argparse
import re
import logging
import time

from dotenv import load_dotenv
from plugin.src.ARAG.recommender import ARAGRecommender 
from plugin.src.ARAG.processing_input import ReviewProcessor


logging.basicConfig(level=logging.INFO)

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


from debug.utils.user2id import load_user_to_idx_map
USER_TO_ID_MAP = {}

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a

class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=None)
        self.processor = ReviewProcessor(target_source=task_set) 

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
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

            elif 'item' in sub_task['description']:
                for item_id in self.task['candidate_list']:
                    item = self.interaction_tool.get_item(item_id=item_id)

                    if item:  
                        # FILTERED ITEMS 
                        keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                        filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                        item_list.append(filtered_item)
                    else:
                        print(f"Warning: No data found for item_id: {item_id}. Skipping.")
                # print(f"Item_list : {item_list}")
            elif 'review' in sub_task['description']:
                all_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                
                candidate_ids = set(self.task['candidate_list'])
                
                filtered_reviews = [
                    r for r in all_reviews 
                    if r.get('item_id') not in candidate_ids
                ]
                
                history_review = str(filtered_reviews)
                
                input_tokens = num_tokens_from_string(history_review)
                if input_tokens > 8000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:8000])
            else:
                pass
        
        self.processor.load_reviews(filtered_reviews)
        self.processor.process_and_split()

        long_term_ctx = self.processor.long_term_context
        current_session = self.processor.short_term_context

        current_idx = USER_TO_ID_MAP.get(self.task['user_id']) 
        final_state = arag_recommender.get_recommendation(
        idx=current_idx,
        task_set=task_set,
        long_term_ctx=long_term_ctx,
        current_session=current_session,
        nli_threshold=4.5,
        candidate_item = item_list )
        
        result = None
        result = final_state['final_rank_list']
        time.sleep(1.5)

        try:
            print('Meta Output:',result)
            return result
        except:
            print('format error')
            return ['']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebSocietySimulator with ARAG Agent Retrie")
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

    parser.add_argument(
        '--model',
        type=str,
        default='classic',
        choices=['classic', 'user_cold_start','item_cold_start'],
        help='Type of scenario to run (classic, user_cold_start,item_cold_start )'
    )
    

    args = parser.parse_args()
    task_set = args.task_set
    scenario = args.scenario

    csv_mapping_path = f"./debug/mappings/mapping_{scenario}_{task_set}.csv"

    " MAPPING FOR DEBUG "
    USER_TO_ID_MAP = load_user_to_idx_map(csv_mapping_path)
    
    " Load Dataset and simulator "
    simulator = Simulator(data_dir="../dataset/output_data_all/", device="gpu", cache=True) 


    " Load scenarios"
    simulator.set_task_and_groundtruth(task_dir=f"../dataset/task/{scenario}/{task_set}/tasks", groundtruth_dir=f"../dataset/task/{scenario}/{task_set}/groundtruth")
    
    " Set Agent"
    simulator.set_agent(MyRecommendationAgent)

    " Set LLM client - CHANGE API KEY "
    load_dotenv()
    " -- OPEN AI -- "
    openai_api_key = os.getenv("OPEN_API_KEY")
    # if args.model == 'open-ai':
    #     model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens = 1000)
    # else :
    #     model = None
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    arag_recommender = ARAGRecommender(model=model, 
        data_base_path=f'../plugin/storage/item_storage_{task_set}')
    
    
    " -- GROQ -- "
    # groq_api_key = os.getenv("GROQ_API_KEY2") # Change API-KEY HERE
    # model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key = os.getenv("GROQ_API_KEY3"))


    " Run evaluation "
    " Note : If you set the number of tasks = None, the simulator will run all tasks."

    " Option 1: No Threading "
    # agent_outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=False)

    " Option 2: Threading - Max_workers = Numbers of Threads"
    agent_outputs = simulator.run_simulation(number_of_tasks=100, enable_threading=False)

    " Evaluate Result "
    evaluation_results = simulator.evaluate()
    with open(f'./results/{scenario}/evaluation_results_ARAG_fixretrieve_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
