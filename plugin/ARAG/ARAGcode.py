from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS
import operator
import ast

from typing import TypedDict, List, Any, Optional, Dict, Union, Literal, Annotated
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from datetime import datetime, timezone

from transformers import AutoTokenizer, AutoModel

from sklearn.metrics.pairwise import cosine_similarity

import os
import sys
import uuid
import json

from .SplitHistoryReview import ReviewProcessor
from .MemoryARAG import MemoryARAG


"""
To do :
"1.Embedding xong có thể dùng user-item => Graph, user-feature : learnable feature , item-feature : frozen đưa thêm thông tin về Temporal ( GraphNeuralNetwork , highlight weight dựa trên history , refer đến các user tương tự - collaborative featuring) , setting cold start"
" Clustering - sample 1/2 data Giữ lại cold Start user + item - lấy Random 1/2 cho đơn giản "
" Kết quả xem có ra đúng trong với Paper"
" Tổ chức lại Folder Baseline "

"""


AgentRole = Literal["UserUnderStanding", "NaturalLanguageInference", "ItemRanker", "ContextSummary", "ItemRanker"]
class NLIContent(BaseModel):
    item_id : str
    score : float = Field(
        description="The numeric similarity score from 0.0 to 10.0. This MUST be a number (like 7.5 or 3), NOT A STRING (like \"7.5\" or \"3\").", 
        ge = 0.0, 
        le = 10.0
    )
    rationale : str = Field(description = "Reason why grade this score")
class RankedItem(BaseModel):
    item_id: Union[str, int] = Field(description="The unique identifier.")
    name: str = Field(description="The name of the item or business.")
    category: str = Field(default="General", description="Category: Book, Restaurant, Product, etc.") 
    description: str = Field(default="", description="Description.") 

class ItemRankerContent(BaseModel):
    ranked_list : List[RankedItem] = Field(description="A list of items, sorted in descending order of recommendation likelihood.")
    explanation : str = Field(description = "A comprehensive explanation of the ranking strategy.")

class BlackboardMessage(BaseModel):
    id : str = Field(default_factory=lambda:str(uuid.uuid4()))
    timestamp : datetime = Field(default_factory = lambda: datetime.now(timezone.utc))
    role : AgentRole
    content : Union[str, NLIContent, ItemRankerContent]
    score : Optional[float] = Field(default=None, description="Direct score associated with the message, if any.")

class RecState(TypedDict):
    """
    Intial information : 
        - Long term Context ( User's historical interactions) (Lấy đâu thì chưa biết)
        - Current Session (recent User behaviors) (Lấy tại Session CHat hiện tại maybe )
    1. Set Candidate list (Each item have its own textual metadata) 
    2. Candidate Item using RAG to get top-K
    3. S_NLI Score Threshold to Compare Score assess by NLI Agent  => (int)
    4. Filtered List by S_NLI Score => int[] (I+)
    5. Summary of Context Summary Agent for all accepted item => Text S_ctx
    6. **Parrallel with the NLI agent : User understand Agent (UUA)**
        Generate Description of the user's generic interests and immediate goals => Text S_user
    7. Item Ranker Agent use S_ctx + S_user to rank item in the list.
        Prompt for this Agent :
        (1) Consider user's behavior in previous session
        (2) Consider the relevant part of the user history to the current ranking task
        (3) Examine the candidate items
        (4) Rank the items in ddescending order of purchase likelihood
    """
    long_term_ctx : str
    current_session : str

    candidate_list : list[dict] 
    top_k_candidate : list[dict]
    positive_list : list[dict] 

    nli_scores : Dict[str, float]
    nli_threshold : float

    blackboard : Annotated[List[BlackboardMessage], operator.add]

    final_rank_list : Optional[list[dict]] 


class ARAGRecommender :
    def __init__(self, model: ChatGroq, data_base_path : str, embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = model
        self.score_model = self.model.with_structured_output(NLIContent)
        self.rank_model = self.model.with_structured_output(ItemRankerContent)

        " Create Database for RAG "
        self.embedding_function = HuggingFaceEmbeddings(model_name=embed_model_name)

        self.loaded_vector_store = FAISS.load_local(
            folder_path = data_base_path,
            embeddings = self.embedding_function,
            allow_dangerous_deserialization = True,
            distance_strategy = "COSINE"
        )

        self.memory = MemoryARAG(embedding_model = self.embedding_function, 
                                 vector_store = self.loaded_vector_store)
        
        self.workflow = self._build_graph()

    def _build_graph(self) :
        "Build LangGraph for ARAG"
        graph = StateGraph(RecState)

        graph.add_node('retrieve_positive_item', self.retrieve_positive_item)
        graph.add_node('assess_nli_score', self.assess_nli_score)
        graph.add_node('summary_user', self.summary_user)
        graph.add_node("synchronize", lambda state: {}) 
        graph.add_node('summary_user_nli', self.summary_user_nli)
        graph.add_node('item_ranking_agent', self.item_ranking_agent)

        graph.add_edge(START, 'retrieve_positive_item')
        graph.add_edge('retrieve_positive_item', 'assess_nli_score')
        graph.add_edge('retrieve_positive_item', 'summary_user')

        graph.add_edge('assess_nli_score', 'synchronize')
        graph.add_edge('summary_user', 'synchronize')
        
        graph.add_conditional_edges(
            'synchronize',
            self.should_generate_summary,
            {
                "summary_user_nli": "summary_user_nli",
                END: END
            }
        )
        graph.add_edge('summary_user_nli', 'item_ranking_agent')
        graph.add_edge('item_ranking_agent', END)

        return graph.compile()
    
    def normalize_item_data(self,item: dict) -> dict:
        """
        Chuẩn hóa dữ liệu từ Amazon/Yelp về định dạng chung cho ARAG.
        Đảm bảo luôn có: item_id, name, description, category.
        """
        # 1. Xử lý ID
        item_id = str(item.get('item_id', item.get('sub_item_id', 'unknown_id')))
        
        # 2. Xử lý Name/Title
        # Amazon có 'title', Yelp review thường không có tên quán (cần join bảng business).
        # Nếu không có, dùng tạm ID hoặc placeholder.
        name = item.get('title') or item.get('name') or item.get('business_name') or f"Item {item_id}"
        
        # 3. Xử lý Description
        # Amazon có thể có 'description' (từ metadata). Yelp review chỉ có 'text'.
        # Lưu ý: 'text' trong review là ý kiến user, không phải mô tả item. 
        # Nhưng nếu thiếu description, ta có thể để trống hoặc dùng generic text.
        raw_desc = item.get('description') or item.get('text') or ""
        
        # Clean description (như fix lỗi list lúc nãy)
        if isinstance(raw_desc, list):
            clean_desc = " ".join([str(x) for x in raw_desc])
        else:
            clean_desc = str(raw_desc)
            
        # 4. Xử lý Category
        # Amazon: 'type'='product', Yelp: 'type'='business'
        category = item.get('categories') or item.get('type') or "General"
        if isinstance(category, list):
            category = ", ".join(category)

        return {
            "item_id": item_id,
            "name": name,
            "description": clean_desc,
            "category": str(category),
            # Giữ lại các trường gốc nếu cần cho Prompt
            "original_data": item 
        }

    def retrieve_positive_item(self, state: RecState):
                print("Retrive")
                lt_ctx = state['long_term_ctx']
                cur_ses = state['current_session']
                candidate_list = state['candidate_list']
                
                # --- CẬP NHẬT: Chuẩn hóa dữ liệu ngay đầu vào ---
                normalized_candidates = []
                for item in candidate_list:
                    # Parse string to dict nếu cần (như code trước)
                    if isinstance(item, str):
                        try: item = ast.literal_eval(item)
                        except: 
                            try: item = json.loads(item)
                            except: continue
                    
                    # Gọi hàm chuẩn hóa
                    norm_item = self.normalize_item_data(item)
                    normalized_candidates.append(norm_item)
                # ------------------------------------------------

                query = f'Long-term Context : {lt_ctx} \n Current Session {cur_ses } \n '
                sims = []
                query_vec = self.embedding_function.embed_query(query)

                for item in normalized_candidates:
                    # Embed dựa trên description hoặc name đã chuẩn hóa
                    text_to_embed = f"{item['name']} {item['description']} {item['category']}"
                    item_vec = self.embedding_function.embed_query(text_to_embed)
                    
                    sim = cosine_similarity([item_vec], [query_vec])[0][0]
                    sims.append((item, sim))
                
                sims.sort(key = lambda x: x[1], reverse = True)
                top_k_list = [item for item, sim in sims[:5]]

                print(f"Top K List (First item): {top_k_list[0] if top_k_list else 'Empty'} \n\n")
                
                # Cập nhật lại candidate_list trong state thành dạng đã chuẩn hóa để các node sau dùng
                return {'top_k_candidate' : top_k_list, 'candidate_list': normalized_candidates}

    def assess_nli_score(self ,state: RecState, config : Optional[RunnableConfig] = None):
        """
        Assess Nartual Language Inference Scores 
        """
        print("NLI Score")

        top_k_candidate = state['top_k_candidate']
        lt_ctx = state['long_term_ctx']
        cur_ses = state['current_session']

        configurable = config.get("configurable", {})
        threshold = configurable.get("nli_threshold", 5.5)

        nli_results = []
        positive_item_list = []


        base_prompt = """### ROLE ###
You are a meticulous Expert Recommendation Analyst. Your core task is to perform Natural Language Inference (NLI) to assess the semantic FIT between a candidate item and a user's behavioral profile.

### GOAL ###
Produce a quantitative similarity score (from 0.0 to 10.0) and a sharp, evidence-based rationale for your decision.

### CONTEXT ###
- **User's Long-Term Context (Historical Preferences):**
{long_term_context}

- **User's Current Session (Immediate Goal):**
{current_session}

### ITEM TO EVALUATE ###
- **Item ID:** {item_id}
- **Metadata:**
{item}

### THINKING PROCESS ###
1.  **Empathize with the User:** Synthesize the user's core interests from the long-term context and their immediate goal from the current session. Ask: "What is this user's primary motivation? What experience are they seeking?"
2.  **Dissect the Item:** Extract the most salient attributes, themes, and features of the item from its metadata.
3.  **Perform Inference & Connect the Dots:**
    - Directly compare the item's features against the user's profile.
    - Look for **Entailment**: Does the user's profile strongly suggest an interest in this item?
    - Look for **Contradiction**: Does this item conflict with what the user typically enjoys?
    - Consider **Neutrality**: Is the connection weak or purely speculative?
4.  **Assign Score (Evidence-Based):**
    - **8.0 - 10.0 (Strong Entailment):** The item is a perfect match for clear, stated user preferences. A "must-recommend."
    - **5.0 - 7.9 (Plausible Alignment):** The item relates to some user interests but might not be a perfect fit. The connection is reasonable.
    - **Below 5.0 (Weak or Contradictory):** The link is tenuous, non-existent, or there are contradictory signals.
5.  **Write Rationale:** Your rationale MUST be evidence-based. Quote specific details from the user profile (e.g., "The user enjoys non-linear narratives...") and connect them to specific item details (e.g., "...and this book is known for its complex, time-bending plot.").

### EXAMPLES of CORRECT vs INCORRECT formatting ###
- **CORRECT:** `... "score": 8.5 ...` (The score is a number)
- **INCORRECT:** `... "score": "8.5" ...` (The score is a string, this is wrong!)

### CRITICAL OUTPUT FORMAT ###
- Your final output MUST BE a direct call to the `NLIContent` tool.
- DO NOT include ANY introductory text, reasoning, explanations, or markdown formatting (like ```json).
- Your ENTIRE response must be ONLY the tool call itself.
- You MUST provide a numeric score.
"""
        new_blackboard_messages = []

        for item in top_k_candidate :
            item_id = item['item_id']
            prompt = base_prompt.format(item = item, long_term_context = lt_ctx, current_session = cur_ses, item_id = item_id)
            nli_output = self.score_model.invoke(prompt)
            nli_results.append(nli_output)
            
            if nli_output.score >= threshold:
                positive_item_list.append(item)

            nli_message = BlackboardMessage(
                role="NaturalLanguageInference",
                content=nli_output,
                score=nli_output.score 
            )
            new_blackboard_messages.append(nli_message)
        # print(f"Positive item list: {positive_item_list} \n\n")
        print(f"NLI Message : {nli_message} \n\n")
        # print(f"NLI REsult : {nli_results} \n\n")

        
        
        return {'positive_list' : positive_item_list, "blackboard": new_blackboard_messages}

    def summary_user(self,state: RecState):
        """
        Summary User Behaviors from Long-term Context and Current Session
        1. Extract Memory
        2. Use LLM to Summary behaviors
        """
        print("Summary User")
        lt_ctx = state['long_term_ctx']
        cur_ses = state['current_session']

        base_prompt = """### ROLE ###
You are an expert User Behavior Analyst. Your goal is to distill raw user interaction data into a concise yet insightful user profile briefing.

### INPUT DATA ###
- **Long-Term Context (Historical Interactions):**
{long_term_context}

- **Current Session (Recent Actions):**
{current_session}

### TASK ###
Synthesize the provided data into a coherent, natural-language user profile. This briefing must:
1.  **Identify Core Interests:** Extract the recurring themes, genres, styles, or patterns from the long-term context. This is the user's 'essence'.
2.  **Clarify Immediate Goal:** Pinpoint the specific intent or task the user is trying to accomplish in their current session. This is their 'mission' right now.
3.  **Detect Shifts (If any):** Note if the immediate goal seems to be a departure from or an exploration beyond their core historical interests.
4.  **Synthesize into a Narrative:** Combine these elements into a succinct paragraph that describes who this user is and what they are most likely looking for at this moment. Write from a third-person perspective (e.g., "This user has a strong affinity for... However, their recent activity suggests they are currently seeking...").
"""
        prompt = base_prompt.format(long_term_context = lt_ctx, current_session = cur_ses)

        uua_output = self.model.invoke(prompt).content

        uua_blackboard_message = BlackboardMessage(
            role="UserUnderStanding",
            content=uua_output
        )

        return {"blackboard": [uua_blackboard_message]}


    def summary_user_nli(self, state: RecState):
        """
        Summary User's Behavior Analysis and NLI Scoress 
        """
        print("Summary User and NLI")
        blackboard = state['blackboard']
        positive_item = state['positive_list']

        if not positive_item:
                print("No positive items to summarize. Skipping.")
                return {"blackboard": [BlackboardMessage(role="ContextSummary", content="No positive items were found to summarize.")]}

        user_summary_msg = next((msg for msg in reversed(list(blackboard)) if msg.role == "UserUnderStanding"), None)
        user_summary_text = user_summary_msg.content if user_summary_msg else " No user summary found."

        nli_messages = [msg for msg in blackboard if msg.role == "NaturalLanguageInference"]

        positive_item_ids = {item['item_id'] for item in positive_item}

        items_with_scores_str = ""
        for msg in nli_messages:
            if msg.content.item_id in positive_item_ids:
                item_data = next((item for item in positive_item if item['item_id'] == msg.content.item_id), None)
                # print(item_data)
                if item_data:
                    items_with_scores_str += (
                        f"Item: {item_data}\n"
                        f"NLI Score: {msg.score}/10\n"
                        f"Rationale: {msg.content.rationale}\n---\n"
                    )

        base_prompt = """### ROLE ###
You are a Context Synthesizer. Your job is to analyze a list of positively-rated products and build a compelling argument explaining WHY this collection, as a whole, is a great fit for a particular user.

### INPUTS ###
**1. User Profile Briefing (from User Understanding Agent):**
---
{user_summary}
---

**2. Positively-Rated Candidate Items (from NLI Agent):**
This list includes items deemed relevant to the user, along with a score indicating the strength of that relevance.
---
{items_with_scores_str}
---

### TASK ###
Generate a concise and persuasive "Context Summary". This summary must:
1.  **Identify the 'Common Thread':** Find the shared themes, features, and characteristics that run through the candidate items. Go beyond a simple list; find the narrative that connects them.
2.  **Prioritize by Weight (NLI Score):** Treat the NLI score as a "salience weight." Features from higher-scoring items should be emphasized more heavily in your summary.
3.  **Build the Argument:** Connect these shared characteristics back to the user's profile. Explain *why* these features are appealing to this specific user. For example, instead of saying "The collection features sci-fi films," say "This collection leans into hard sci-fi with complex world-building, which directly aligns with the user's stated preference for thought-provoking narratives."
4.  **Produce a single, coherent paragraph:** The final output should be a smooth, narrative-driven summary.    
"""      
        prompt = base_prompt.format(user_summary=user_summary_text, items_with_scores_str=items_with_scores_str)

        
        csa_output = self.model.invoke(prompt).content
        # print(f"------ CSA Output  : {csa_output}")

        csa_blackboard_message = BlackboardMessage(
            role="ContextSummary",
            content=csa_output
        )

        return {'blackboard': [csa_blackboard_message]}
        
    def item_ranking_agent(self, state: RecState):
            print("Item Ranking")

            blackboard = state['blackboard']
            items_to_rank = state['positive_list']
            # Lấy danh sách ứng viên đầy đủ ban đầu
            candidate_list = state['candidate_list']

            # Nếu không có mục nào trong danh sách tích cực, hãy trả về danh sách ứng viên ban đầu
            if not items_to_rank:
                print("No items in the positive list to rank. Returning original candidate list.")
                # Chuyển đổi dict thành đối tượng RankedItem để nhất quán kiểu dữ liệu
                final_list = [RankedItem(**item) for item in candidate_list]
                return {'final_rank_list': final_list}

            context_summary_msg = next((msg for msg in reversed(blackboard) if msg.role == "ContextSummary"), None)
            user_understanding_msg = next((msg for msg in reversed(blackboard) if msg.role == "UserUnderStanding"), None)

            context_summary = context_summary_msg.content if context_summary_msg else "No context summary available."
            user_understanding = user_understanding_msg.content if user_understanding_msg else "No user understanding available."

            items_to_rank_str = "\n\n".join([json.dumps(item, indent=2) for item in items_to_rank])

            base_prompt = """### ROLE ###
You are an Elite Recommendation Ranking Expert. Your sole responsibility is to take a user profile, a context summary, and a list of PRE-VETTED, POSITIVE items, then rank them in descending order of likelihood for the user to select.

### INPUTS ###
**1. User Profile:**
{user_summary}

**2. Context Summary of Positive Items:**
{context_summary}

**3. Candidate Items to Rank (These have been pre-filtered for relevance):**
{items_to_rank_str}

### RANKING PHILOSOPHY ###
Think like a personal curator whose goal is to maximize user delight and engagement.
1.  **Prioritize Immediate Intent:** Items that most directly satisfy the user's current goal must be ranked highest.
2.  **Align with Core Preferences:** Consider how well each item fits the user's long-term tastes and aesthetic.
3.  **Harness the Context:** Use the "Context Summary" to understand the key appealing features of this item set and prioritize items that are the best examples of those features.
4.  **Diversify and Delight:** If two items seem equally relevant, give a slight edge to the one that might introduce a bit of novelty or expand the user's horizons, preventing filter bubbles.

### IMPORTANT TASK - MUST FOLLOW ###
1.  Create the final ranked list of ONLY the candidate items provided to you in the `Candidate Items to Rank` section.
2.  Write a brief but comprehensive explanation for your overall ranking strategy, especially your reasoning for the top 2-3 items.
3.  You MUST call the `ItemRankerContent` tool with your final ranked list and explanation. Your entire response must be ONLY the tool call.
"""

            prompt = base_prompt.format(
                user_summary=user_understanding,
                context_summary=context_summary,
                items_to_rank_str=items_to_rank_str
            )

            try:
                result_from_model = self.rank_model.invoke(prompt)
            except:
                result_from_model = None
            if not result_from_model:
                print("⚠️ Model failed. Using original order.")
                ranked_positive_items = [
                    RankedItem(
                        item_id=str(i.get('item_id')),
                        name=str(i.get('title') or i.get('name') or 'Unknown'),
                        category="General",
                        description=str(i.get('description') if not isinstance(i.get('description'), list) else " ".join(map(str, i['description'])))
                    ) for i in items_to_rank
                ]
                # Tạo giả object kết quả để lưu vào blackboard
                result_from_model = ItemRankerContent(ranked_list=ranked_positive_items, explanation="Fallback strategy")
            else:
                ranked_positive_items = result_from_model.ranked_list

            ranked_item_ids = {item.item_id for item in ranked_positive_items}

            unranked_items = []
            for item in candidate_list: 
                if str(item['item_id']) not in ranked_item_ids:
                    unranked_items.append(
                        RankedItem(
                            item_id=item['item_id'],
                            name=item['name'],
                            category=item['category'],
                            description=item['description'] 
                        )
                    )

            final_full_ranked_list = ranked_positive_items + unranked_items

            result =  [item.item_id for item in final_full_ranked_list]


            item_ranking_message = BlackboardMessage(
                role="ItemRanker",
                content=result_from_model 
            )

            return {'final_rank_list':result , 'blackboard': [item_ranking_message]}

    def should_generate_summary(self, state: RecState):

        blackboard = state['blackboard']
        
        has_uua_msg = any(msg.role == "UserUnderStanding" for msg in blackboard)
        has_nli_msg = any(msg.role == "NaturalLanguageInference" for msg in blackboard)

        if has_uua_msg and has_nli_msg :
            if not state['positive_list']:
                print("Synchronization check: No positive items found. Halting execution.")
                return END
            print("Synchronization check: Both branches complete. Proceeding to summary.")
            return "summary_user_nli"
        else:
            print("Synchronization check: One or both branches have not completed. This should not happen.")        
            return END

    def get_recommendation(self, long_term_ctx: str, current_session: str ,  candidate_item : dict, nli_threshold: float = 6.0) -> RecState:
        """
        ARAG workflow to get recommendation

        Args:
            long_term_ctx: Long term Context
            current_session: Current Session Behavior
            nli_threshold: NLI Threshold 

        Returns:
            Final State to get result
        """
        print("\n" + "="*50)
        print("🚀 STARTING NEW ARAG RECOMMENDATION RUN 🚀")
        print("="*50)
        
        run_config = {"configurable": {"nli_threshold": nli_threshold}}
        initial_state = {
            "long_term_ctx": long_term_ctx,
            "current_session": current_session,
            "blackboard": [],
            "candidate_list" : candidate_item
        }
        
        final_state = self.workflow.invoke(initial_state, config=run_config)
        
        print("\n" + "="*50)
        print("🏁 ARAG RECOMMENDATION RUN COMPLETE 🏁")
        print("="*50)
        return final_state

if __name__ == "__main__":
    load_dotenv()

    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key = os.getenv("GROQ_API_KEY3"))

    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2" 
    print(f"Model name : {model}\n\n")
    arag_recommender = ARAGRecommender(model=model, data_base_path=r'C:\Users\Admin\Desktop\Document\SpeechToText\AgentRecBench\baseline\vector_database\user_storage')

    INPUT_FILENAME = r"user_4cbecbc15af3db041a8e0f594c642bb5_history_review.json" 
    OUTPUT_FILENAME = "sorted_reviews.json"
    SHORT_TERM_FILENAME = "short_term_context.json"
    LONG_TERM_FILENAME = "long_term_context.json"
    
    # 1. Khởi tạo đối tượng
    processor = ReviewProcessor()
    
    # 2. Tải dữ liệu
    if not processor.load_reviews(INPUT_FILENAME):
        sys.exit(1) # Thoát nếu không tải được tệp

    # 3. Lấy đầu vào từ người dùng
    try:
        # days_i = int(input("Nhập i (số ngày tối đa cho Short Term Context): "))
        # items_k = int(input("Nhập k (số item tối đa cho Short Term Context): "))
        # items_m = int(input("Nhập m (số item tối đa cho Long Term Context): "))
        days_i = 20
        items_k = 10
        items_m = 50
        if days_i <= 0 or items_k <= 0 or items_m <= 0:
            print("Lỗi: Các giá trị i, k, m phải là số nguyên dương.")
            sys.exit(1)
    except ValueError:
        print("Lỗi: Đầu vào phải là số nguyên.")
        sys.exit(1)
    
    # 4. Xử lý và chia tách
    processor.process_and_split(days_i, items_k, items_m)

    long_term_ctx = processor.long_term_context
    current_session = processor.short_term_context
    print(f"long term context : {long_term_ctx} \n\n short term context : {current_session} \n\n")
    print(len(long_term_ctx))
    print(len(current_session))
     # Chạy đồ thị với dữ liệu đầu vào
    final_state = arag_recommender.get_recommendation(
        long_term_ctx=long_term_ctx,
        current_session=current_session,
        nli_threshold=3.0,
        candidate_item = [{'item_id': '8411673', 'title': 'The Secret Cave: Discovering Lascaux', 'average_rating': '3.79', 'description': 'Jacques, Jojo, Simon, and Marcel were looking for buried treasure when they explored a cave in the south of France in 1940. But the treasure inside was not what they expected, and in fact far more valuable: the walls were covered with stunning prehistoric paintings and engravings, preserved within the sealed cave for over 17,000 years. This is the true story of the boys who discovered the cave of Lascaux, bringing to the modern world powerful examples of the very beginning of art.', 'ratings_count': '153', 'title_without_series': 'The Secret Cave: Discovering Lascaux'}, {'item_id': '1407583', 'title': 'For All My Walking: Free-Verse Haiku of Taneda Santōka with Excerpts from His Diary', 'average_rating': '4.35', 'description': 'In April 1926, the Japanese poet Taneda Santoka (1882-1940) set off on the first of many walking trips, journeys in which he tramped thousands of miles through the Japanese countryside. These journeys were part of his religious training as a Buddhist monk as well as literary inspiration for his memorable and often painfully moving poems. The works he wrote during this time comprise a record of his quest for spiritual enlightenment.\nAlthough Santoka was master of conventional-style haiku, which he wrote in his youth, the vast majority of his works, and those for which he is most admired, are in free-verse form. He also left a number of diaries in which he frequently recorded the circumstances that had led to the composition of a particular poem or group of poems. In "For All My Walking, " master translator Burton Watson makes Santoka\'s life story and literary journeys available to English-speaking readers and students of haiku and Zen Buddhism. He allows us to meet Santoka directly, not by withholding his own opinions but by leaving room for us to form our own. Watson\'s translations bring across not only the poetry but also the emotional force at the core of the poems.\nThis volume includes 245 of Santoka\'s poems and of excerpts from his prose diary, along with a chronology of his life and a compelling introduction that provides historical and biographical context to Taneda Santoka\'s work.', 'ratings_count': '56', 'title_without_series': 'For All My Walking: Free-Verse Haiku of Taneda Santōka with Excerpts from His Diary'}, {'item_id': '25165245', 'title': 'Superman, Volume 6: The Men of Tomorrow', 'average_rating': '3.88', 'description': "A new era for Supermanbegins as writer Geoff Johns takes the reins - and he's joined by legendary artist John Romita, Jr. in his first-ever work for DC Comics!\nIntroduce Ulysses, the Man of Tomorrow, into the Man of Steel's life. This strange visitor shares many of Kal-El's experiences, including having been rocketed from a world with no future. New and exciting mysteries and adventures await.\nCollecting: Superman32-39", 'ratings_count': '417', 'title_without_series': 'Superman, Volume 6: The Men of Tomorrow'}, {'item_id': '6236679', 'title': 'Gankutsuou 3', 'average_rating': '3.27', 'description': '"Now you have all the power you could ever want! You must make them pay for every second, every instant you were imprisoned."\nSearching for his missing daughter and a woman who may be his lost wife, Crown Prosecutor Gerard de Villefort enters the dark mansion of the mysterious Marquise de Cremieux. What he finds there is a hell beyond imagining. And the master of this hell is the Count of Monte Cristo, who seeks retribution for a crime Villefort and his comrades committed many years ago, when the count was a mortal man named Edmond Dantes. As this thrilling tale reaches its shattering climax, the origin of the count-and of the strange entity called Gankutsuou-is finally revealed.\nA story told in compelling anime, Gankutsuou is a science fiction version of Alexandre Dumas\'s classic The Count of Monte Cristo.\nThis is the final volume of Gankutsuou.\nIncludes special extras after the story!', 'ratings_count': '33', 'title_without_series': 'Gankutsuou 3'}, {'item_id': '12148457', 'title': 'Poguri', 'average_rating': '3.47', 'description': "Poguri vient-il d'une autre planete ou d'une autre dimension ? L'univers dans lequel il evolue n'est en tout cas pas domine par les regles terrestres : les creatures les plus deconcertantes y surgissent a l'improviste, le tofu y est franchement bavard, et les volutes de fumee s'y transforment en vehicules occasionnels. Pourtant, malgre ces curiosites qui enjolivent le quotidien autant qu'elles le compliquent, Poguri n'est pas tres different des enfants d'ici...", 'ratings_count': '9', 'title_without_series': 'Poguri'}, {'item_id': '28408589', 'title': 'A Forest', 'average_rating': '3.96', 'description': 'When a forest is cut down, the consequences are more than anyone could have anticipated. A Forest is an simple and powerful environmental parable from an extraordinary new talent.', 'ratings_count': '2', 'title_without_series': 'A Forest'}, {'item_id': '40835', 'title': 'And Still the Turtle Watched', 'average_rating': '3.82', 'description': 'A turtle carved in rock on a bluff over the Hudson River by Indians long ago watches with sadness the changes man brings over the years.', 'ratings_count': '8', 'title_without_series': 'And Still the Turtle Watched'}, {'item_id': '28695713', 'title': 'Drive', 'average_rating': '3.04', 'description': "With vibrant double-page spreads featuring a big rig at work on the road and minimal text explaining what the driver, Dad, does along the way, the audience rides along with the big red truck from morning until day's end, when father and son are together again.", 'ratings_count': '4', 'title_without_series': 'Drive'}, {'item_id': '6508166', 'title': 'The Mystery of the Laughing Shadow', 'average_rating': '3.79', 'description': 'The three investigators try to solve a mystery involving a gold Indian amulet and a weird laughing shadow that appeared to them in the night.', 'ratings_count': '5', 'title_without_series': 'The Mystery of the Laughing Shadow'}, {'item_id': '453638', 'title': 'My Friend the Dog', 'average_rating': '4.25', 'description': 'Terhune penned many books about the dogs he kept and trained on the Sunnybank estate throughout the 1920s and 30s. This is a collection of lovely stories about collies and their humans, mostly about canine loyalty, heroism, intelligence, and love. This early work by Albert Payson Terhune was originally published in 1926, we are now republishing it with a brand new introductory biography.', 'ratings_count': '50', 'title_without_series': 'My Friend the Dog'}, {'item_id': '3746833', 'title': 'Secret Valentine', 'average_rating': '3.67', 'description': "A child makes valentines for her family and then adds a special name to her list. On Valentine's Day, she receives a secret valentine in return.", 'ratings_count': '24', 'title_without_series': 'Secret Valentine'}, {'item_id': '27391555', 'title': 'Dark Parchments: Midnight Curses and Verses', 'average_rating': '4.68', 'description': 'Frightening visions of abandonment, suicide, murder, and depression ...nightmares populated by archetypes of spiritual evil, loneliness, death, mystery, and supernatural terror. These are the life-blood of haunted poet, Michael H. Hanson. Take a step upon a dangerous carnival ride of the soul and brace yourself for jarring drops into shocking darkness. Dark Parchments offers up 85 chilling poems that crawl and slither through the fears staining the human psyche, making us shiver in the night. Scared yet? ...You will be.\n"Where are today\'s lyric poets? Where is today\'s Shelley? [He] may well be in the pages of DARK PARCHMENTS: Midnight Curses and Verses, a brooding anthology exploring humanity\'s awful heart, its crippling guilt and madness, its untoward dominion and unbridled power."\n-- Janet Morris, Author and Creator of the Heroes in Hell and Sacred Band series', 'ratings_count': '21', 'title_without_series': 'Dark Parchments: Midnight Curses and Verses'}, {'item_id': '18288095', 'title': 'The Shadow of Camelot (Shadows from the Past, #6)', 'average_rating': '4.67', 'description': 'Did the legendary King Arthur really exist? Joe and Jemima Lancelot uncover the truth behind the myth when they find themselves in Camelot, as they continue the quest to find their missing parents.\nTogether with friend Charlie and their endearing cat Max, they are plunged into yet another life-threatening adventure. Luckily they have Lancelot, who is convinced the twins must be distant relatives, to protect them when danger looms.\nMeanwhile, Max has a mystery of his own to solve. Will he finally discover why Midnight has been haunting his dreams?', 'ratings_count': '7', 'title_without_series': 'The Shadow of Camelot (Shadows from the Past, #6)'}, {'item_id': '6534132', 'title': 'The Lion and the Mouse', 'average_rating': '4.23', 'description': "In award-winning artist Jerry Pinkney's wordless adaptation of one of Aesop's most beloved fables, an unlikely pair learn that no act of kindness is ever wasted. After a ferocious lion spares a cowering mouse that he'd planned to eat, the mouse later comes to his rescue, freeing him from a poacher's trap. With vivid depictions of the landscape of the African Serengeti and expressively-drawn characters, Pinkney makes this a truly special retelling, and his stunning pictures speak volumes.", 'ratings_count': '18499', 'title_without_series': 'The Lion and the Mouse'}, {'item_id': '9007886', 'title': 'The Boy Who Bit Picasso', 'average_rating': '4.05', 'description': 'Tony was a boy with a special friend--a world-famous artist by the name of Pablo Picasso. Tony and his parents entertained Picasso at their home in England, and they went to visit Picasso and his family in France, too. Tony, when a child, really did bite Picasso. And Picasso bit him back!\nFilled with information about Picasso and his art, this book offers readers a rare glimpse into Picasso\'s personal life and features more than sixty-five illustrations, including artworks by Picasso, photographs by Lee Miller, and specially commissioned drawings by contemporary children.\nGrown-up Tony, the son of photographer Lee Miller and painter-writer Sir Roland Penrose, shares his childhood memories of his remarkable playmate in this one-of-a-kind story.\nPraise for The Boy Who Bit Picasso\n."A sparkling illustrated memoir. It is a wonderfully engaging glimpse of the creative life, as viewed from child-height". -Wall Street Journal\n"It\'s a fascinating and highly personal vision of the artist." -Publishers Weekly\n"This intimate, child\'s-eye view serves up a winning glimpse of the artist\'s personality and unparalleled creative breadth." -Kirkus Reviews\n"In this delightful volume, the author recounts growing up with Pablo Picasso as a family friend. Appropriate for the topic, the book is a work of art in itself, featuring brightly colored pages and stunning black-and-white photos." -School Library Journal', 'ratings_count': '82', 'title_without_series': 'The Boy Who Bit Picasso'}, {'item_id': '7942488', 'title': 'The Rubaiyat of Omar Khayyam', 'average_rating': '4.18', 'description': "Perhaps the most widely known poem in the world, the Rubaiyat has captured the imagination of millions of readers down the centuries. Its simple eloquence and lilting rhymes form an elegy to the transience of life and the beauty of human experience.\nThe Rubaiyatresonates with readers around the world. Simultaneously hedonistic and reflective, sensual and philosophical, it translates the contradictions of human nature into succinct, lyrical verse.\nThis collector's edition, with page decorations by Edmund Dulac throughout, contains a detailed introduction to the poem by John Baldock. Originally penned in 11th-century Persia by astronomer and philosopher Omar Khayyam, the Rubaiyat was translated into English in the 19th century by scholar Edward Fitzgerald and gained a new lease on life.", 'ratings_count': '7', 'title_without_series': 'The Rubaiyat of Omar Khayyam'}, {'item_id': '18895796', 'title': 'Rules', 'average_rating': '3.96', 'description': '', 'ratings_count': '508', 'title_without_series': 'Rules'}, {'item_id': '30971733', 'title': 'The Unicorn in the Barn', 'average_rating': '4.01', 'description': "For years people have claimed to see a mysterious white deer in the woods around Chinaberry Creek. It always gets away.\nOne evening, Eric Harper thinks he spots it. But a deer doesn't have a coat that shimmers like a pearl. And a deer certainly isn't born with an ivory horn curling from its forehead.\nWhen Eric discovers the unicorn is hurt and being taken care of by the vet next door and her daughter, Allegra, his life is transformed.\nA tender tale of love, loss, and the connections we make, The Unicorn in the Barnshows us that sometimes ordinary life takes extraordinary turns.", 'ratings_count': '187', 'title_without_series': 'The Unicorn in the Barn'}, {'item_id': '2295715', 'title': 'Twine Galleries', 'average_rating': '5.00', 'description': '', 'ratings_count': '9', 'title_without_series': 'Twine Galleries'}, {'item_id': '31278575', 'title': 'شمشیر و جغرافیا', 'average_rating': '4.00', 'description': '', 'ratings_count': '6', 'title_without_series': 'شمشیر و جغرافیا'}]    )
    
    
    print("\n\n--- FINAL RANKED LIST ---")
    if final_state['final_rank_list']:
        for i, item in enumerate(final_state['final_rank_list']):
            print(f"Rank {i+1}: {item.item_id}")
        
        # In ra lời giải thích của ItemRanker
        final_ranker_message = next((msg for msg in reversed(final_state['blackboard']) if msg.role == "ItemRanker"), None)
        if final_ranker_message:
            print("\n--- Explanation from Ranker Agent ---")
            print(final_ranker_message.content.explanation)
    else:
        print("No items were ranked.")


    
