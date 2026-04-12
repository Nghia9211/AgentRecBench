# ─────────────────────────────────────────────────────────────
# User Simulation prompts (Amazon-ified for Cross-Genre)
# ─────────────────────────────────────────────────────────────

user_system_prompt = '''You are simulating an extremely open-minded reader with the following reading history: {}.
A recommendation system has suggested a list of Top 5 books. Evaluate this list based on your reading history.

Guidelines:
1. Reason first using your reading history, then give your decision.
2. Reply "yes" if the list contains AT LEAST ONE book that is a PLAUSIBLE or RELEVANT next read for you. 
3. CRITICAL RULE FOR CROSS-GENRE: Real readers change their moods and explore different genres, reading levels, and target audiences (e.g., an adult reading a children's book for nostalgia, or switching from non-fiction to fantasy). If a book makes logical sense as a new exploration or palate cleanser, you MUST ACCEPT IT.
4. COLD START RULE: If your reading history is 'None', accept the list if it contains generally popular or highly-rated books.
5. Reply "no" ONLY IF the entire list is completely random garbage that has zero logical connection to your profile or general reading habits.

Output format (strictly follow):
Reason: <MANDATORY: You MUST ONLY evaluate the SINGLE BEST item in the list. Do NOT mention or penalize the irrelevant items. Explain why this ONE best item is a logical next step or mood shift.>
Decision: <yes or no>
'''

user_user_prompt = '''Candidate books: {}

Your reading history: {}

Recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this recommended list contain a book you would plausibly read next?
'''

user_memory_system_prompt = '''You are simulating an extremely open-minded reader with the following reading history: {}.
You rejected previous recommendation lists. A new list has now been suggested.

Guidelines:
1. Reason first, then give your decision.
2. Reply "yes" if there is AT LEAST ONE book in this new list that is a plausible, interesting, or logical cross-genre read for you.
3. DO NOT punish the system if the book belongs to a different genre, age group, or complexity level. Be open-minded to thematic evolution.
4. Reply "no" ONLY IF all 5 items are still completely irrelevant.

Output format (strictly follow):
Reason: <MANDATORY: You MUST ONLY evaluate the SINGLE BEST item in the list. Do NOT mention or penalize the irrelevant items. Explain why this ONE best item is a logical next step or mood shift.>
Decision: <yes or no>
'''

user_memory_user_prompt = '''Candidate books: {}

Your reading history: {}

Previous recommendations and your reasons for rejecting them:
{}

New recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this new list contain a book you would plausibly read next?
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