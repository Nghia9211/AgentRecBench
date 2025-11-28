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
from plugin.ARAG.ARAGcode import ARAGRecommender ,RecState, BlackboardMessage, ItemRankerContent ,RankedItem , NLIContent
from plugin.ARAG.SplitHistoryReview import ReviewProcessor
logging.basicConfig(level=logging.INFO)

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

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
                        # FILTERED ITEMS #oGDGlUbOjHxmmCh8ZYcDCg
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
                if input_tokens > 12000:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    history_review = encoding.decode(encoding.encode(history_review)[:12000])
            else:
                pass
        

        processor.process_and_split()

        long_term_ctx = processor.long_term_context
        current_session = processor.short_term_context

        final_state = arag_recommender.get_recommendation(
        long_term_ctx=long_term_ctx,
        current_session=current_session,
        nli_threshold=2.0,
        candidate_item = item_list )
        
        result = None
        result = final_state['final_rank_list']
       
        try:
            print('Meta Output:',result)
            return result
        except:
            print('format error')
            return ['']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebSocietySimulator with DummyAgent")
    parser.add_argument(
        '--task_set', 
        type=str, 
        default='amazon', 
        choices=['amazon', 'yelp', 'goodreads'],
        help='Name of the dataset to use (amazon, yelp, goodreads)'
    )
    

    args = parser.parse_args()
    task_set = args.task_set
    
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
    openai_api_key = os.getenv("OPEN_API_KEY")
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens = 1000)

    processor = ReviewProcessor(target_source = task_set)
    arag_recommender = ARAGRecommender(model=model, data_base_path=f'C:/Users/Admin/Desktop/Document/SpeechToText/RecSystemCode/storage/item_storage_{task_set}')
    
    " -- GROQ -- "
    # groq_api_key = os.getenv("GROQ_API_KEY2") # Change API-KEY HERE
    # model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key = os.getenv("GROQ_API_KEY3"))


    " Run evaluation "
    " Note : If you set the number of tasks = None, the simulator will run all tasks."

    " Option 1: No Threading "
    # agent_outputs = simulator.run_simulation(number_of_tasks=2, enable_threading=False)

    " Option 2: Threading - Max_workers = Numbers of Threads"
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers = 5)

    " Evaluate Result "
    evaluation_results = simulator.evaluate()
    with open(f'./results/user_coldstart/evaluation_results_ARAG_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")
