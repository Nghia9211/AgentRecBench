import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
import sys
import argparse
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from websocietysimulator.llm import LLMBase, InfinigenceLLM, GroqLLM , OpenAILLM
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from langchain.docstore.document import Document
from websocietysimulator import Simulator
from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOT
import re
import logging
import time
from langchain.embeddings import HuggingFaceEmbeddings
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

class MyRecommendationAgent(RecommendationAgent):
    """
    Participant's implementation of SimulationAgent
    """
    def __init__(self, llm:LLMBase):
        super().__init__(llm=llm)
        self.reasoning = ReasoningCOT(profile_type_prompt='', memory=None,llm=self.llm)
        "--- IF NOT HAVE OPENAI EMBEDDING ---"
        self.memory = MemoryDILU(llm=self.llm)

        " --- IF USE OPENAI EMBEDDING, UNCOMMENT THIS --- "
        # openai_api_key = os.getenv("OPEN_API_KEY")
        # sync_client = OpenAI(api_key=openai_api_key)
        # sync_embeddings = OpenAIEmbeddings(client=sync_client.embeddings)
        # self.memory = MemoryDILU(llm=self.llm, embedding_model = sync_embeddings)
        " --- IF USE OPENAI, UNCOMMENT THIS --- "

        


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
                        # FILTERED ITEMS #
                        keys_to_extract = ['item_id', 'name','stars','review_count','attributes','title', 'average_rating', 'rating_number','description','ratings_count','title_without_series']
                        filtered_item = {key: item[key] for key in keys_to_extract if key in item}
                        item_list.append(filtered_item)
                    else:
                        print(f"Warning: No data found for item_id: {item_id}. Skipping.")
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
        retrieved_memory = ""
        if filtered_reviews:  # This checks that the list is not None and not empty
            for i, his in enumerate(filtered_reviews): 
                current_trajectory = str(his)
                self.memory.addMemory(current_trajectory)

            query_scenario = f"History review of {self.task['user_id']}"
            retrieved_memory = self.memory.retriveMemory(query_scenario)
            # print(f"Retrieved Memory : {retrieved_memory} \n\n")

        task_description = f"""
You are a recommendation agent. Your task is to recommend items for a user based on their profile, historical reviews, and a list of candidate items.

Here is the information you have:

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

Do not include any other text, explanations, or markdown formatting around the list. Your response should only contain the list itself.
"""
        
        result = self.reasoning(task_description)
        
        final_recommendation = []
        try:
            print('Meta Output:',result)
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group()
            else:
                print("No list found.")
            
            final_recommendation = eval(result)
            print('Processed Output:', final_recommendation)
            # time.sleep(4)
        except:
            print('format error')
            final_recommendation = ['']

        return final_recommendation


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
    simulator = Simulator(data_dir=f"../dataset/output_data_all/", device="gpu", cache=True) 


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
    simulator.set_llm(OpenAILLM(api_key=openai_api_key))

    " -- GROQ -- "
    # groq_api_key = os.getenv("GROQ_API_KEY3") # Change API-KEY HERE
    # simulator.set_llm(GroqLLM(api_key = groq_api_key ,model="meta-llama/llama-4-scout-17b-16e-instruct"))


    " Run evaluation "
    " Note : If you set the number of tasks = None, the simulator will run all tasks."

    " Option 1: No Threading "
    # agent_outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=False)

    " Option 2: Threading - Max_workers = Numbers of Threads"
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers = 10)

    " Evaluate Result "
    evaluation_results = simulator.evaluate()
    with open(f'./results/user_coldstart/evaluation_results_CoTMemoryAgent_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"The evaluation_results is :{evaluation_results}")