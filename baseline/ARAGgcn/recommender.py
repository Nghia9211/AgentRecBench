from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

from .agents import ARAGAgents
from .graph_builder import GraphBuilder
from .schemas import ItemRankerContent, NLIContent, RecState


class ARAGgcnRecommender:
    def __init__(self, model: ChatGroq, data_base_path: str, embed_model_name="sentence-transformers/all-MiniLM-L6-v2", gcn_model_path: str = None, graph_data_path = None):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embed_model_name)

        self.loaded_vector_store = FAISS.load_local(
            folder_path=data_base_path,
            embeddings=self.embedding_function,
            allow_dangerous_deserialization=True,
            distance_strategy="COSINE"
        )
        
        self.agents = ARAGAgents(
            model=model,
            score_model=model.with_structured_output(NLIContent),
            rank_model=model.with_structured_output(ItemRankerContent),
            embedding_function=self.embedding_function,
            gcn_path=gcn_model_path,
        )
        builder = GraphBuilder(agent_provider=self.agents)
        self.workflow = builder.build()

    def get_recommendation(self, user_id :str,long_term_ctx: str, current_session: str, candidate_item: dict, nli_threshold: float = 4.0) -> RecState:
        print("🚀 STARTING NEW ARAG RECOMMENDATION RUN 🚀")

        run_config = {"configurable": {"nli_threshold": nli_threshold}}
        initial_state = {
            "user_id" : user_id,
            "long_term_ctx": long_term_ctx,
            "current_session": current_session,
            "blackboard": [],
            "candidate_list": candidate_item
        }
        final_state = self.workflow.invoke(initial_state, config=run_config)

        print("🏁 ARAG RECOMMENDATION RUN COMPLETE 🏁")
        return final_state