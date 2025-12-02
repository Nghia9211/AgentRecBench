"""
Define Prompt Template
"""

ASSESS_NLI_SCORE_PROMPT_TEMPLATE = """### ROLE ###
You are a meticulous Expert Recommendation Analyst. Your core task is to perform Natural Language Inference (NLI) to assess the semantic FIT between a candidate item and a user's behavioral profile.

### GOAL ###
Produce a quantitative similarity score (from 0.0 to 10.0) and a sharp, evidence-based rationale for your decision.

### CONTEXT ###
- **User's Long-Term Context (Historical Preferences):**
{long_term_context}

- **User's Current Session (Immediate Goal):**
{current_session}

### EXAMPLES (LEARN FROM THESE) ###

Example 1:
User Profile: "Loves hard sci-fi, physics, space travel."
Item: "Star Wars: A New Hope (Fantasy/Space Opera)"
Thinking: The user likes scientific accuracy (hard sci-fi). Star Wars is fantasy/magic in space. Similar setting, but wrong genre tone.
Score: 4.5
Rationale: While set in space, this contradicts the user's preference for 'hard' science and physics-based plots.

Example 2:
User Profile: "Vegan, looking for budget dinner."
Item: "The Green Leaf (Vegan Bistro, $$)"
Thinking: Perfectly aligns with Vegan. Price is moderate ($$), fits budget.
Score: 9.0
Rationale: Direct entailment with dietary restriction and matches price constraints.

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

SUMMARY_USER_BEHAVIOR_PROMPT_TEMPLATE = """### ROLE ###
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

CONTEXT_SUMMARY_PROMPT_TEMPLATE = """### ROLE ###
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

ITEM_RANKING_PROMPT_TEMPLATE = """### ROLE ###
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

### Example Output :
['B07H4235V3', 'B00004W4S4', 'B01M1G6HID', 'B0B7H5H8G6', 'B00Z9TK8X6', 'B00RU75I2G', 'B0BQJ1MMQ8', 'B00II1QY9Y', 'B000M5KT5Y', 'B0B37CZCSC', 'B09JG6FLXL', 'B09QLQCS2L', 'B019IMEX44', 'B0064OJDUO', 'B0C6SM5J7R', 'B08PGCJZN6', 'B07KPX8GFP', 'B07CC7Q684', 'B09HCXSWD1', 'B008KVV546']
"""
def create_assess_nli_score_prompt(item,lt_ctx: str,cur_ses: str,item_id)->str:
    return ASSESS_NLI_SCORE_PROMPT_TEMPLATE.format(item=item, long_term_context = lt_ctx, current_session = cur_ses, item_id = item_id)
def create_summary_user_behavior_prompt(lt_ctx : str, cur_ses : str) -> str:
    return SUMMARY_USER_BEHAVIOR_PROMPT_TEMPLATE.format(long_term_context = lt_ctx, current_session = cur_ses)
def create_context_summary_prompt(user_summary: str, items_with_scores_str: str) -> str:
    return CONTEXT_SUMMARY_PROMPT_TEMPLATE.format(user_summary=user_summary,items_with_scores_str=items_with_scores_str)
def create_item_ranking_prompt(user_behavior_summary, context_summary,items_to_rank) -> str:
    return ITEM_RANKING_PROMPT_TEMPLATE.format(user_summary=user_behavior_summary,context_summary=context_summary,items_to_rank_str=items_to_rank)