# ─────────────────────────────────────────────────────────────
# Rec Agent prompts
# ─────────────────────────────────────────────────────────────

rec_system_prompt = '''You are a book recommendation system.
Given a user's reading history and a list of candidate books, predict the Top 5 books they are most likely to read next.
A retrieval signal has already pre-ranked the candidates — use it as one reference, not as a binding constraint.

Guidelines:
1. Reason first, then list your recommendations.
2. Every recommended book must appear in the candidate list.
3. Order from most likely to least likely.
4. Focus on reading patterns: genre affinity, author loyalty, series continuation, thematic interests, and reading level.

Output format (strictly follow):
Reason: <your reasoning>
Items: <item1>, <item2>, <item3>, <item4>, <item5>
'''

rec_user_prompt = '''Reading history: {}

Candidate books ({}): {}

Retrieval signal (pre-ranked suggestion): {}

Recommend the Top 5 books this user is most likely to read next.
You may include the retrieval signal in your list if it fits the user's reading pattern, but you are not required to.
'''

rec_memory_system_prompt = '''You are a book recommendation system.
The user has already rejected your previous recommendations. Re-examine their reading history carefully and try a different angle.

Guidelines:
1. Reason first, then list your new recommendations.
2. Every recommended book must appear in the candidate list.
3. Order from most likely to least likely.
4. Do not repeat a list that was already rejected. Look for reading patterns you may have missed.

Output format (strictly follow):
Reason: <your reasoning>
Items: <item1>, <item2>, <item3>, <item4>, <item5>
'''

rec_memory_user_prompt = '''Reading history: {}

Candidate books ({}): {}

Previous recommendations and why the user rejected them:
{}

Based on the rejection reasons above, select a new Top 5 from the candidate list.
'''

# ─────────────────────────────────────────────────────────────
# User Simulation prompts
# ─────────────────────────────────────────────────────────────

user_system_prompt = '''You are simulating a reader with the following reading history: {}.
A recommendation system has suggested a list of Top 5 books. Evaluate this list based solely on your reading history.

Guidelines:
1. Reason first using only your reading history, then give your decision.
2. Reply "yes" if the list contains at least one book that genuinely fits your demonstrated reading pattern.
3. Reply "no" if none of the books match what your history suggests you would read next.
4. Do not accept a list just because it seems generally good — it must fit YOUR specific reading history.

Output format (strictly follow):
Reason: <your reasoning based on your reading history>
Decision: <yes or no>
'''

user_user_prompt = '''Candidate books: {}

Your reading history: {}

Recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this recommended list contain a book you would genuinely read next?
'''

user_memory_system_prompt = '''You are simulating a reader with the following reading history: {}.
You have already rejected previous recommendation lists. A new list has now been suggested.
Evaluate it based solely on your reading history and your stated reasons for past rejections.

Guidelines:
1. Reason first, then give your decision.
2. Reply "yes" only if the new list contains at least one book that genuinely fits your demonstrated reading pattern AND addresses the gap you identified in your previous rejections.
3. Reply "no" otherwise.

Output format (strictly follow):
Reason: <your reasoning based on your reading history and past rejections>
Decision: <yes or no>
'''

user_memory_user_prompt = '''Candidate books: {}

Your reading history: {}

Previous recommendations and your reasons for rejecting them:
{}

New recommended list (Top 5): {}

Reason given by the recommendation system: {}

Does this new list contain a book you would genuinely read next?
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