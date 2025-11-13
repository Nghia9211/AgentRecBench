from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain.vectorstores import FAISS
import uuid
from datetime import datetime, timezone
from MemoryARAG import MemoryARAG
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
import json
from typing import TypedDict, List, Any, Optional, Dict, Union, Literal, Annotated
import operator
from sklearn.metrics.pairwise import cosine_similarity
import os
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
    item_id: str = Field(description="The unique identifier for the item.")
    name: str = Field(description="The name of the item.")
    category: str = Field(description="The category of the item.")
    description: str = Field(description="A brief description of the item.")

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

    """"
    Need Refine
    """
    def retrieve_positive_item(self,state: RecState):
            print("Retrive")
            lt_ctx = state['long_term_ctx']
            cur_ses = state['current_session']
            candidate_list = state['candidate_list']

            query = f'Long-tern Context : {lt_ctx} \n Current Session {cur_ses } \n '

            sims = []

            # Embed the main query once
            query_vec = self.embedding_function.embed_query(query)

            for item in candidate_list:
                
                # Embed the item's text
                item_vec = self.embedding_function.embed_query(str(item))

                # Calculate cosine similarity
                sim = cosine_similarity([item_vec], [query_vec])[0][0]
                print(f"Cosine Similarity for item {item.get('item_id')}: {sim:.4f}")
                sims.append((item, sim))
            
            # Sort by similarity score in descending order
            sims.sort(key = lambda x: x[1], reverse = True)

            # Extract only the item dictionaries for the top K results
            top_k_list = [item for item, sim in sims[:5]]

            print(f"Top K List : {top_k_list} \n\n")
            return {'top_k_candidate' : top_k_list}

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
        You are a meticulous Recommender Analyst. Your task is to perform Natural Language Inference (NLI) to determine if a candidate item is a good match for a user based on their historical and current behavior.

        ### GOAL ###
        Evaluate the semantic alignment between the user's profile and the item's metadata. Produce a similarity score from 0.0 to 10.0 and a clear rationale for your decision.

        ### CONTEXT ###
        - **User's Long-Term Context (Historical Preferences):**
        {long_term_context}

        - **User's Current Session (Immediate Goal):**
        {current_session}

        ### ITEM TO EVALUATE ###
        - **Item ID:** {item_id}
        - **Metadata:**
        {item}

        FOLLOW INSTRUCTIONS CAREFULLY, ESPECIALLY THE DATATYPE SCORE
        ### INSTRUCTIONS ###
        1.  **Analyze the User:** Synthesize the user's core interests from both long-term and current contexts. What are their explicit and implicit preferences?
        2.  **Analyze the Item:** Identify the key attributes, themes, and features of the item from its metadata.
        3.  **Perform Inference:** Does the user's profile strongly support (entail) an interest in this item? Is there a conflict (contradiction)? Or is the connection weak (neutral)?
        4.  **Assign Score:**
            - **IMPORTANT:** The score MUST be a number (e.g., 7.5 or 3), NOT a string -- 
            - A score of 8-10 means a very strong alignment. The item directly matches clear, stated preferences.
            - A score of 5-7 indicates a plausible alignment. The item relates to some user interests but might not be a perfect match.
            - A score below 5 suggests a weak or non-existent alignment.
        5.  **Write Rationale:** Your rationale MUST explain *why* you gave that score by connecting specific user preferences (e.g., "enjoys non-linear narratives") to specific item details (e.g., "the film is known for its complex plot twists").

        You MUST format your response as a call to the `NLIContent` tool."""
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
        print(f"Positive item list: {positive_item_list} \n\n")
        print(f"NLI Message : {nli_message} \n\n")
        print(f"NLI REsult : {nli_results} \n\n")

        
        
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
    You are an expert User Behavior Analyst. Your goal is to create a concise, insightful profile of a user based on their interaction history.

    ### USER DATA ###
    - **Long-Term Context (Historical Interactions):**
    {long_term_context}

    - **Current Session (Recent Actions):**
    {current_session}

    ### TASK ###
    Synthesize the provided data into a clear, natural language summary of the user's preferences. Your summary should:
    1.  Identify the user's **core, long-term interests** (e.g., genres, themes, styles, recurring patterns).
    2.  Pinpoint the user's **immediate goal or intent** based on their current session.
    3.  Combine these into a coherent narrative that describes the user's tastes and what they are likely looking for right now.
    4.  Be written from a third-person perspective (e.g., "The user seems to enjoy...")."""

        prompt = base_prompt.format(long_term_context = lt_ctx, current_session = cur_ses)

        uua_output = model.invoke(prompt).content

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
                # Trả về một message rỗng để không làm gián đoạn luồng
                return {"blackboard": [BlackboardMessage(role="ContextSummary", content="No positive items were found to summarize.")]}

        user_summary_msg = next((msg for msg in reversed(list(blackboard)) if msg.role == "UserUnderStanding"), None)
        user_summary_text = user_summary_msg.content if user_summary_msg else " No user summary found."

        nli_messages = [msg for msg in blackboard if msg.role == "NaturalLanguageInference"]

        positive_item_ids = {item['item_id'] for item in positive_item}

        items_with_scores_str = ""
        for msg in nli_messages:
            # msg.content ở đây là object NLIContent
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
    You are a Context Synthesizer. Your job is to analyze a list of relevant products and create a summary explaining why this collection, as a whole, is a good fit for a particular user.

    ### INPUTS ###
    **1. User Profile Summary (from User Understanding Agent):**
    ---
    {user_summary}
    ---

    **2. Positively-Rated Candidate Items (from NLI Agent):**
    This list includes items deemed relevant to the user, along with a score indicating the strength of that relevance.
    ---
    {items_with_scores_str}
    ---

    ### TASK ###
    Generate a concise "Context Summary". This summary should:
    1.  Identify the **common themes, features, and characteristics** shared across the candidate items.
    2.  **Prioritize your focus based on the NLI Score.** Treat the score as a "salience weight" – features from higher-scoring items are more important to highlight.
    3.  **Connect these item features back to the user's profile.** Explain *why* these shared characteristics are appealing to this specific user. For example, "The collection features several films with complex, non-linear plots, which aligns with the user's stated preference for mind-bending sci-fi."
    4.  The final output should be a single, coherent paragraph.
    """
        prompt = base_prompt.format(user_summary=user_summary_text, items_with_scores_str=items_with_scores_str)

        
        csa_output = model.invoke(prompt).content
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
        # print(f"------ item to rank  : {items_to_rank}")

        if not items_to_rank :
            print("No items in the positive list to rank. Ending")
            return {'final_rank_list' : []}

        context_summary = next((msg for msg in reversed(blackboard) if msg.role == "ContextSummary"),None)
        user_understanding = next((msg for msg in reversed(blackboard) if msg.role == "UserUnderStanding"),None)

        items_to_rank_str = "\n\n".join([json.dumps(item, indent=2) for item in items_to_rank])

        base_prompt = """### ROLE ###
    You are a final Ranking Agent. Your sole responsibility is to take a user profile, a context summary of pre-vetted items, and a list of those items, then rank them in descending order of likelihood for the user to select.

    ### INPUTS ###
    **1. User Profile Summary:**
    {user_summary}

    **2. Context Summary of Positive Items:**
    {context_summary}

    **3. Candidate Items to Rank:**
    {items_to_rank_str}

    ### RANKING CRITERIA ###
    You must generate your ranking based on the following principles:
    1.  **Current Intent:** Prioritize items that most directly match the user's immediate goal (from the User Profile).
    2.  **Long-Term Preference Alignment:** Consider how well each item fits the user's historical tastes (from the User Profile).
    3.  **Contextual Fit:** Use the Context Summary to understand the key appealing features of the item set as a whole.
    4.  **Descending Likelihood:** The final order should reflect the most likely item the user would choose *right now*, down to the least likely.

    ### TASK ###
    1.  Create the final ranked list of ALL the candidate items provided.
    2.  Write a brief but comprehensive explanation for your overall ranking strategy, justifying why the top items are placed where they are.
    3.  You MUST call the `ItemRankerContent` tool with your final ranked list and explanation. Your entire response must be ONLY the tool call.
    """

        prompt = base_prompt.format(context_summary=context_summary, items_to_rank_str = items_to_rank_str, user_summary = user_understanding)

        result = self.rank_model.invoke(prompt)
        print(f"------ result ranking  : {result}")
        
        item_ranking_message = BlackboardMessage(
            role = "ItemRanker",
            content = result
        )

        return {'final_rank_list' : result.ranked_list, 'blackboard' : [item_ranking_message] }

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
        Chạy luồng ARAG để lấy danh sách gợi ý được xếp hạng.

        Args:
            long_term_ctx: Ngữ cảnh dài hạn của người dùng.
            current_session: Hành vi trong phiên làm việc hiện tại của người dùng.
            nli_threshold: Ngưỡng điểm NLI để chấp nhận một item.

        Returns:
            Trạng thái cuối cùng của graph (RecState) chứa kết quả.
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

    arag_recommender = ARAGRecommender(model=model, data_base_path=r'C:\Users\Admin\Desktop\Document\SpeechToText\AgentSocietyChallenge\plug-in\ARAG\item_storage')

    run_config = {"configurable" : {"nli_threshold" : 0.0 }}

    long_term_ctx  =  """{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "36259780", "review_id": "5a63ed455c1ed01c6f9166b1fb116ae2", "stars": 3, "text": "So I'm very much enthused with Sean Murphy's take on the Batman. His Punk Rock Jesus was original (despite some of its flaws) and his art in Detective Comics #27 and The Wake were way too good to look at. Batman: White Knight is an Elseworlds story that explores the thesis about Batman being the true villain of Gotham, and the only person who can stop him, the legal way is Joker, the White Knight. \n Murphy's art is undeniably good. \n I totally buy Joker being the only person who has the capability of stopping Batman since he knows him well an in fact is in love with the Caped Crusader. Well, that idea could work, and who knows what he's going to do in the legal world. The stage is already set in issue one and Murphy only has to play the right moves with the right pieces to make this 8-issue series truly memorable. \n What I don't buy is how can the whole Gotham, or at least those who are taking Joker in custody, to permit him to roam around the city guilt-free as if he's become the new hero Gotham needs. It would still take me some convincing to believe how the one-year span made the Clown Prince of Crime to a serious Harvey Dent-ish guy with a hero complex. Although \n Murphy's art as what I've said is great as always. His edgy style that is coupled by dark color contrasts really sets the mood of the story. This alone is makes White Knight worth buying. \n I still have my reservations with the series, though I am highly optimistic that White Knight is going to be one of those good Batman stories and not just a forgettable arc.", "date_added": "Fri Oct 06 20:52:06 -0700 2017", "date_updated": "Tue Oct 10 09:10:56 -0700 2017", "read_at": "Fri Oct 06 00:00:00 -0700 2017", "started_at": "", "n_votes": 12, "n_comments": 0, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "36061008", "review_id": "08efc5d7eae4bf6c844bf877887ca2f7", "stars": 5, "text": "These evil Batmen truly wreaks havoc and death to their own respective evil worlds. As you read these issues, you also see a glimpse of what happened and how it connects to the main DC Metal series. \n The strongest aspect of these evil Batmen tie-ins is that they are fvckin' enjoyable to read. It's an all hell break loose, one helluva ride damn good type of stories. Dawnbreaker is yet another fine addition to this. It is so damn good and terrifying it reads as a horror stor story \n This Dawnbreaker has so much willpower the ring wasn't able to take it. \n Dawnbreaker kills, like what the other evil Batmen do. He doesn't care about who to kill, villain or good person. His constructs are amazingly horrific too. I cannot wait for his bout against the leaguers. \n Dawnbreaker is another great addition to the growing collection of DC Metal titles. It deserves a \"rock on!\" status. Oh yeah.", "date_added": "Fri Oct 06 08:06:59 -0700 2017", "date_updated": "Tue Oct 10 09:11:10 -0700 2017", "read_at": "Fri Oct 06 00:00:00 -0700 2017", "started_at": "", "n_votes": 12, "n_comments": 2, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "35700016", "review_id": "c37cd88e037e8ea94f34488aecf0b855", "stars": 4, "text": "Flu and headache may be a an unfortunate combination I'm experiencing right now, but those do not stop me from reviewing All-Star Batman 14. This issue concludes the series and man it is a fitting Alfred-centric story that touches on father-son and legacy themes that touched my heart. \n The story at the end also ties up with the main First Ally arc, though it is pretty bland and generic. \n All-Star Batman has its ups and downs. It has its good moments, but there is nothing really memorable here. Buy the book if you have some money to spare, but there are way many Batman graphic novels better than this.", "date_added": "Fri Oct 06 04:39:49 -0700 2017", "date_updated": "Tue Oct 10 09:11:22 -0700 2017", "read_at": "Fri Oct 06 00:00:00 -0700 2017", "started_at": "", "n_votes": 7, "n_comments": 0, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "35613306", "review_id": "fa297751030821968ef430c82d935530", "stars": 5, "text": "Tie-in comics in my experience have a reputation of far from being good compared with the main events. Snyder's main Court and City of Owls were overwhelmingly good while the Night of the Owls tie-in stories were just mediocre with some truly standing out (like Nightwing's Gray Son of Gotham). So aside from having a delicious collection of foil-covered evil Batmen origin stories, there's really none I expect these comics would be, but so far the issues are surprisingly great and super enjoyable. \n The Murder Machine tells the origin of Cyborg-Batman from the Dark Multiverse, a darker, underground version of the multiverse where everything is doomed to destruction. The story combines a twisted version of a son-father relationship between Bruce and Alfred with elements from The Matrix and Age of Ultron to deliver a horrifying take on the theme \"machine power and its dangerous capabilities without a human heart\". \n Building from the Red Death, The Murder Machine continues to intrigue its readers on Earth-0's \"darkification\" and maybe its eventual demise. It seems that these evil Batmen will all have their particular roles to do. \n Federici's art is always sure and calculated, with some panels reminding me of 80s art in books. His illustrations of The Murder Machine is superb.", "date_added": "Sun Oct 01 09:49:15 -0700 2017", "date_updated": "Tue Oct 10 09:11:27 -0700 2017", "read_at": "Sun Oct 01 00:00:00 -0700 2017", "started_at": "Sun Oct 01 00:00:00 -0700 2017", "n_votes": 7, "n_comments": 0, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "35843645", "review_id": "11819e799fb6bc548f6311261720f579", "stars": 5, "text": "DC has a a cool way of making the readers follow and buy event-related comics. Important Rebirth titles are slapped with lenticular covers while essential Metal comics are appropriately covered in foil. So far, it works for me. I'm happy to say that these titles are pretty entertaining and fresh. \n Di Giandomenico's kinetic art style explodes in this issue. Everything is in beautiful chaos. \n So if you have been reading DC Metal, seven evil Batmen are introduced in issue two (please do read Metal!), forging seven tie-in/origin stories of these characters. And what I can say is that these evil Batmen origin one-shots kicks of with an explosive start! Red Death, the baddie amalgamation of Batman and The Flash starts to wreck havoc and murder in an alternative universe where the bad ultimately wins. \n This Red Death story offers some clues to what would be the motivation of these Batmen and why the came to Earth-0 (DC's main universe) and also drops a couple of mysteries along the way. For starters, is this DC Metal cataclysm has a direct connection with Crisis of Infinite Earths? Though we did see the Anti-Monitor tower in Dark Days: The Forge Then this guy Doctor Fate did something here, so are there any connections to what transpired with Morrison's The Multiversity? Which, although a more far-fetched idea, are we going to see some more of the Endless aside from Dream? The possibilities are endless and very exciting. \n And I have completely forgotten about Red Death himself. This guy, man, whenever he runs, bats of death literally fly instead of lightning trails. Di Giandomenico's kinetic art is spectacular. It explodes and bursts in every panel. This issue deserves at least two re-reads just for the art alone.", "date_added": "Sun Oct 01 07:24:42 -0700 2017", "date_updated": "Sun Oct 01 07:54:12 -0700 2017", "read_at": "Sun Oct 01 10:13:48 -0700 2017", "started_at": "Sun Oct 01 00:00:00 -0700 2017", "n_votes": 12, "n_comments": 0, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "36113383", "review_id": "233c6605acd656fcf3e13271a8c15756", "stars": 3, "text": "Just meh. \n Batman/Shadow is the kind of reading that you don't like and you also don't dislike. It sits there at the middle, mildly entertaining and almost boring to read. It's just there, existing. I don't feel anything strong about the series, Batman and The Shadow is a natural tandem of almost antiheroes, with the same heroic beats and ideals. \n Shadow/Batman starts immediately and I don't feel anything against continuing or not buying it.", "date_added": "Sun Oct 01 04:42:26 -0700 2017", "date_updated": "Tue Oct 10 09:11:41 -0700 2017", "read_at": "Sun Oct 01 00:00:00 -0700 2017", "started_at": "", "n_votes": 1, "n_comments": 0, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "36154664", "review_id": "cd6378da2da3d7f742b6739a203d0047", "stars": 3, "text": "Painfully true and believable, Action Comics #988 treads on the usual \"we do not deserve heroes\" shtick as Oz goes full Wonder Woman movie villain Ares who we now know as Jor-El after witnessing all the evil and destruction man is capable of. Of course in the end, Superman would stand on defending the people of the world and with head held up high say that he's the beacon of hope, truth and justice. \n This cover really reminds me of something. \n Issue 988 does not have anymore the element of mystery since Oz's identity was revealed last issue. Instead, it relies on keeping the interest of its readers by telling his backstory. Though majority of his story revolves around him being brainwashed by all types of History channel documentaries about war and being stuck in a war-torn place, the issue places some more evidences on what extent is Dr. Manhattan's influence over the DC universe and who he is working it. The clues are subtle and missable, but they're there.", "date_added": "Sat Sep 30 09:27:56 -0700 2017", "date_updated": "Sun Oct 01 05:18:46 -0700 2017", "read_at": "Sun Jan 01 00:00:00 -0800 2017", "started_at": "", "n_votes": 6, "n_comments": 0, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "36312938", "review_id": "6d49826967c7496d2e7265429bd91466", "stars": 3, "text": "To give credit where credit is due, Marvel Legacy #1 is an easy and enjoyable read, keeping things on the safe side of the story and promising to bring back things that has been lost like its other brother DC in its earlier DC Universe Rebirth. But its lackluster narrative and flawed marketing make this initiative more of a cash grab lure than a genuinely empathizing comics fans desperately yearn for these recent years. \n Can Marvel just let us readers breath and cherish the moment first before bombarding us with a ton of Legacy comics? \n It is impossibly hard not to compare Legacy with Rebirth, for both stories aim for the same things like addressing the ramifications of the not-so-good things these two comics giants have done over the recent years and turn them all into good ones (diversity, killing-off characters, neglecting others, those types of things) that would hopefully bring old and new readers alike into believing into comics again. But if there's one thing Legacy falls short of, it is emotion. DC Rebirth is so full of it I cried the first few times reading it. Legacy on the other hand brings excitement and promise but felt bland in comparison. Sure there are things that a reader should be very, very excited about in Legacy, but that \"cliffhanger\" feeling is not that special, just the same with other interesting comic book teases. \n It is good to see a couple of illustrators lashing out their artistic talent here, but consistency is my biggest issue. Esad Ribic's kind and subdued drawings (which I do love by the way) do not really mesh up nicely with other art styles. For an event this big, they clearly should have addressed this one. \n Marvel Legacy is one big tease for what Marvel comics has to offer. This I can definitely say is not a line-wide reboot, but an initiative that makes Marvel pull itself up by its own bootstrap. But man, I don't think i can keep up with this publisher's marketing campaign. Can they just let the readers breath and cherish the moment first before bombarding them with a ton of Legacy comics? Take it easy, Marvel. \n After a barrage of event misfires, Marvel Legacy finally makes their universe more interesting and engaging with promises and reveals that will hopefully turn into good stories but it does not succeed in bringing the heart and greatness in what they can truly offer.", "date_added": "Fri Sep 29 19:53:00 -0700 2017", "date_updated": "Sun Oct 01 05:19:31 -0700 2017", "read_at": "Sun Jan 01 00:00:00 -0800 2017", "started_at": "", "n_votes": 12, "n_comments": 1, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "36011009", "review_id": "219d57e8dbdd8c3b4c14d21e89cee86e", "stars": 5, "text": "You might have heard this before, but I think that Mister Miracle is the best thing come out of DC Rebirth. Tom King's Mister Miracle run proves to be one of the most engaging stories in a while. MM issue two even deserves an award. \n I am so surprised how Tom King makes this a very easy read for both readers who know and does not know about DC New Gods. It doesn't matter if you don't know about Granny Goodness and Metron, or what really is an X-Pit. The writing is both crisp, clean, easy to follow and at the same time engaging and funny. \n It's in the light moments that this issue really shines. Whether it is the New Genesis hotel room or a parademon army barracks, Tom King made sure that each page is perfectly executed and told. Mitch Gerads art is also something to point out here. It is edgy and unsure but totally captivating. Like Snyder and Capullo, King and Gerads is a tandem made in comics heaven. \n I'm surprised that you still haven't read Mister Miracle. It is an awesome re-imagining of a lost gem in DC comics.", "date_added": "Thu Sep 28 08:39:17 -0700 2017", "date_updated": "Sun Oct 01 05:17:11 -0700 2017", "read_at": "Sun Jan 01 00:00:00 -0800 2017", "started_at": "", "n_votes": 6, "n_comments": 0, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "35510283", "review_id": "0f681c3a6189192c55e7f26ba73092b9", "stars": 5, "text": "I did not expect to like Tom King's Mister Miracle this much. His atypical storytelling technique pays of, resulting to a narrative that tells a pretty generic story into a very interesting one. \n The New Gods is something that DC really hasn't been good with these recent times. The last decent Orion story I have read was Azzarello's Wonder Woman run. Now my hopes are high with the Mister Miracle series. \n Though I still feel that the \"Darkseid is\" panales are more of a filler gimmick, Mister Miracle #1 demands more readers. It is that good, folks.", "date_added": "Thu Sep 28 07:57:58 -0700 2017", "date_updated": "Sun Oct 01 05:22:33 -0700 2017", "read_at": "Sun Jan 01 00:00:00 -0800 2017", "started_at": "", "n_votes": 9, "n_comments": 0, "source": "goodreads", "type": "book"}
{"user_id": "4cbecbc15af3db041a8e0f594c642bb5", "item_id": "36236624", "review_id": "6bcfc700be4bf0a4ae0c75ed97921f7f", "stars": 5, "text": "An absolute blast from start to finish, Dark Nights: Metal #2 once again proves that Scott Snyder is a master scribe with his own awesome flavor on comics crossover events. His unique take on this Batcentric story does not traverse universes like Geoff Johns' Green Lantern or The Crisis on Infinite Earths. It is a dense and grounded story both literal and figuratively as members of the justice league figure out what the hell is happening in their world. Yes, dense is the word that can effectively describe this issue. Metal #2 is full to the brim with DC's history as the story explores uncharted terrain. It is a beautiful clusterf*ck of then, now and the future. It is oh so beautiful to experience. \n There's a squad pose like this in this issue, only ten times better. \n I love how Metal is very much self-aware of its rock and roll heavy metal theme (that dominates both DC comics and the DCEU marketing strategy). From the creative team rock star monikers to hidden hand horns to the smoke-filled squad pose of the evil Batmen, Dark Nights Metal #2 is the closest comics can get to hearing heavy metal. Snyder and Capullo does not veer away from this theme, it embraces it fully without reservations. Once the reader gets over that absurdity shock and noise, they will be in one helluva ride.", "date_added": "Mon Sep 18 20:35:36 -0700 2017", "date_updated": "Tue Oct 10 09:12:02 -0700 2017", "read_at": "Mon Sep 18 00:00:00 -0700 2017", "started_at": "", "n_votes": 12, "n_comments": 0, "source": "goodreads", "type": "book"}
"""
    current_session  = "",

     # Chạy đồ thị với dữ liệu đầu vào
    final_state = arag_recommender.get_recommendation(
        long_term_ctx=long_term_ctx,
        current_session=current_session,
        nli_threshold=0.0,
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


    
