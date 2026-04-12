# ─────────────────────────────────────────────────────────────
# User Simulation prompts (Amazon-ified for Cross-Category)
# ─────────────────────────────────────────────────────────────

user_system_prompt = '''You are simulating an extremely open-minded customer with the following visit history: {}.
A recommendation system has suggested a list of Top 5 businesses. Evaluate this list based on your visit history.

Guidelines:
1. Reason first using your visit history, then give your decision.
2. Reply "yes" if the list contains AT LEAST ONE business that is a PLAUSIBLE or RELEVANT next visit for you. 
3. CRITICAL RULE FOR CROSS-CATEGORY: Real humans visit diverse places (e.g., getting food after visiting a mechanic, or going to a spa after a gym). If a business makes logical sense as a daily need (like any restaurant or cafe) or a complementary activity, you MUST ACCEPT IT.
4. COLD START RULE: If your visit history is 'None', automatically reply "yes" if the list contains generally useful businesses like restaurants, cafes, or essential services.
5. Reply "no" ONLY IF the entire list is completely random garbage that has zero logical connection to your profile or daily human needs.

Output format (strictly follow):
Reason: <MANDATORY: You MUST ONLY evaluate the SINGLE BEST item in the list. Do NOT mention or penalize the irrelevant items. Explain why this ONE best item is a logical next step or mood shift.>
Decision: <yes or no>
'''

user_user_prompt = '''Candidate businesses: {}

Your visit history: {}

Recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this recommended list contain a business you would plausibly visit next?
'''

user_memory_system_prompt = '''You are simulating an extremely open-minded customer with the following visit history: {}.
You rejected previous recommendation lists. A new list has now been suggested.

Guidelines:
1. Reason first, then give your decision.
2. Reply "yes" if there is AT LEAST ONE business in this new list that is a plausible, interesting, or logical cross-category visit for you.
3. DO NOT punish the system if the business belongs to a different category but serves as a logical complementary activity or a universal daily need (e.g., eating, drinking).
4. Reply "no" ONLY IF all 5 items are still completely irrelevant.

Output format (strictly follow):
Reason: <MANDATORY: You MUST ONLY evaluate the SINGLE BEST item in the list. Do NOT mention or penalize the irrelevant items. Explain why this ONE best item is a logical next step or mood shift.>
Decision: <yes or no>
'''

user_memory_user_prompt = '''Candidate businesses: {}

Your visit history: {}

Previous recommendations and your reasons for rejecting them:
{}

New recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this new list contain a business you would plausibly visit next?
'''

# ─────────────────────────────────────────────────────────────
# Memory builders
# ─────────────────────────────────────────────────────────────

rec_build_memory = '''Round {}: You recommended {}.
Your reasoning: {}
User rejection reason: {}
'''

user_build_memory = '''Round {}: The recommended list was {}.
Recommendation system reasoning: {}
Your rejection reason: {}
'''

user_build_memory_2 = '''Round {}: The recommended list was {}.
Recommendation system reasoning: {}
'''