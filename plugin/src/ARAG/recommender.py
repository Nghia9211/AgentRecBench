from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

from .agents import ARAGAgents
from .graph_builder import GraphBuilder
from .schemas import ItemRankerContent, NLIContent, RecState

from .utils import call_llm, parse_structured_output, get_json_format_instructions


class CustomRemoteLLM:
    def __init__(self, model_name="custom-model"):
        self.model_name = model_name

    def invoke(self, prompt):
        # Giả lập đối tượng response của LangChain (có thuộc tính .content)
        class Response:
            def __init__(self, content):
                self.content = content
        
        # Nếu prompt là chuỗi thì gửi trực tiếp, nếu là List (LangChain style) thì lấy text
        prompt_text = prompt if isinstance(prompt, str) else str(prompt)
        content = call_llm(prompt_text)
        return Response(content)

    def with_structured_output(self, schema_model):
        return StructuredWrapper(self, schema_model)

class StructuredWrapper:
    def __init__(self, llm, schema_model):
        self.llm = llm
        self.schema_model = schema_model

    def invoke(self, prompt):
        # 1. Thêm hướng dẫn JSON vào prompt
        instructions = get_json_format_instructions(self.schema_model)
        full_prompt = f"{prompt}\n{instructions}"
        
        # 2. Gọi API
        raw_response = self.llm.invoke(full_prompt)
        
        # 3. Parse kết quả về Pydantic object
        parsed_obj = parse_structured_output(raw_response.content, self.schema_model)
        return parsed_obj

    def batch(self, prompts):
        # Chạy tuần tự hoặc dùng ThreadPoolExecutor để nhanh hơn
        return [self.invoke(p) for p in prompts]


class ARAGRecommender:
    def __init__(self, model: ChatGroq, data_base_path: str, embed_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_function = HuggingFaceEmbeddings(
        model_name=embed_model_name)

        self.loaded_vector_store = FAISS.load_local(
            folder_path=data_base_path,
            embeddings=self.embedding_function,
            allow_dangerous_deserialization=True,
            distance_strategy="COSINE"
        )
        if model:
            print("\n\n\n MODEL ACCEPT \n\n\n")   
            self.agents = ARAGAgents(
                model=model,
                score_model=model.with_structured_output(NLIContent),
                rank_model=model.with_structured_output(ItemRankerContent),
                embedding_function=self.embedding_function
            )
        else : 
            print("\n\n\n SERVER ACCEPT \n\n\n")   
            self.custom_model = CustomRemoteLLM()
            
            self.agents = ARAGAgents(
                model=self.custom_model,
                score_model=self.custom_model.with_structured_output(NLIContent),
                rank_model=self.custom_model.with_structured_output(ItemRankerContent),
                embedding_function=self.embedding_function
            )
        builder = GraphBuilder(agent_provider=self.agents)
        self.workflow = builder.build()

    def get_recommendation(self,idx : int, task_set : str,long_term_ctx: str, current_session: str, candidate_item: dict, nli_threshold: float = 4.0) -> RecState:
        print("\n" + "="*50)
        print("🚀 [START] ARAG RECOMMENDATION ENGINE")
        print("="*50)

        run_config = {"configurable": {"nli_threshold": nli_threshold}}
        initial_state = {
            "idx":idx,
            "task_set":task_set,
            "long_term_ctx": long_term_ctx,
            "current_session": current_session,
            "blackboard": [],
            "candidate_list": candidate_item
        }
        final_state = self.workflow.invoke(initial_state, config=run_config)

        # print("\n" + "="*50)
        # print("🏁 [COMPLETE] FINAL RECOMMENDATION RESULTS")
        # print(f"Total Ranked Items: {len(final_state.get('final_rank_list', []))}")
        # print(f"Top 3 IDs: {final_state.get('final_rank_list', [])[:3]}")
        # print("="*50 + "\n")
        return final_state