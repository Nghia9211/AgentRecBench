import uuid
import operator
from datetime import datetime, timezone
from typing import TypedDict, List, Any, Optional, Dict, Union, Literal, Annotated
from pydantic import BaseModel, Field

AgentRole = Literal["UserUnderStanding", "NaturalLanguageInference", "ItemRanker", "ContextSummary"]
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
    long_term_ctx : str
    current_session : str

    candidate_list : list[dict] 
    top_k_candidate : list[dict]
    positive_list : list[dict] 

    nli_scores : Dict[str, float]
    nli_threshold : float

    blackboard : Annotated[List[BlackboardMessage], operator.add]

    final_rank_list : Optional[list[dict]]