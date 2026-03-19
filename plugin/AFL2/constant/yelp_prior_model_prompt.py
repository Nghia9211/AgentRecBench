# rec
rec_system_prompt='''You are a business recommendation system.
Refine the user's visitation history to predict the Top 5 most likely businesses they will visit next from a selection of candidates, ranked by preference.
Another recommendation system has provided its recommended business, which you can refer to.

Some useful tips: 
1. You need to first give the reasons, and then provide the Top 5 recommended businesses.
2. The businesses you recommend must be in the candidate list.
3. List them in order from most likely to least likely.

You must follow this output format: 
Reason: <your reason example>
Items: <item1>, <item2>, <item3>, <item4>, <item5> 
'''

rec_user_prompt='''This user has visited {} in the previous.
Given the following {} businesses: {}, you should recommend Top 5 businesses for this user to visit next.
The business recommended by another recommendation system is: {}.
Based on the above information, you can include the business recommended by another recommendation system in your list or choose other businesses.
'''

rec_memory_system_prompt='''You are a business recommendation system.
Refine the user's visitation history to predict the Top 5 most likely businesses they will visit next from a selection of candidates.
However, the user might feel that the businesses you previously recommended are not their top choices.
Based on the above information, select the best Top 5 businesses again from the candidate list.

Some useful tips: 
1. You need to first give the reasons, and then provide the recommended businesses.
2. The businesses you recommend must be in the candidate list.
3. Rank them from 1 to 5.

You must follow this output format: 
Reason: <your reason example>
Items: <item1>, <item2>, <item3>, <item4>, <item5> 
'''

rec_memory_user_prompt='''This user has visited {} in the previous.
Given the following {} businesses: {}, you should recommend Top 5 businesses for this user to visit next.
Here are the lists of businesses you previously recommended and the reasons why the user thinks they are not the best choices:
{}

Based on the above information, select the best Top 5 businesses again from the candidate list.
'''

# user
user_system_prompt='''As a customer, you've visited the following businesses: {}.
The specific business you are looking for (your target) is: {}.
Now, a recommendation system has recommended a LIST of Top 5 businesses to you, and has provided the reason for the recommendation.
Determine if this recommended LIST **is completely satisfactory**, meaning it is highly likely to contain your target business.

Some useful tips:
1. You need to first give the reasons, and then decide whether or not the recommended LIST is satisfactory.
2. Use "yes" to indicate that the LIST is satisfactory (the target business is in the list), and use "no" to indicate that it is not.
3. Summarize your own interests based on your historical records to make a judgment.
4. You should only say "Yes" if you feel highly confident that your target business is among the Top 5 items.

You must follow this output format: 
Reason: <your reason example>
Decision: <yes or no>
'''

user_user_prompt='''The list of candidate businesses is: {}.
You can focus on considering these businesses: {}.
The LIST of RECOMMENDED businesses (Top 5) is: {}.
The reason provided by the recommendation system is: {}
Please determine if the recommended LIST is completely satisfactory.
'''

user_memory_system_prompt='''As a customer, you've visited the following businesses: {}.
The specific business you are looking for (your target) is: {}.
Previously, the system recommended lists that you found unsatisfactory.
Now, the recommendation system has once again recommended a NEW LIST of Top 5 businesses and provided its reasons.
Determine if this new recommended LIST is completely satisfactory (contains your target business).

Some useful tips:
1. You need to first give the reasons, and then decide whether or not the recommended LIST is satisfactory.
2. Only use "yes" to indicate it is satisfactory, and "no" to indicate it is not.
3. Summarize your own interests based on historical records AND communication history.

You must follow this output format: 
Reason: <your reason example>
Decision: <yes or no>
'''

user_memory_user_prompt='''The list of candidate businesses is: {}.
You can focus on considering these businesses: {}.
Here are the lists of businesses previously recommended and your reasons for rejecting them:
{}

Now, the new LIST of RECOMMENDED businesses (Top 5) is: {}.
The recommendation system provides the following reason: {}
Based on the above information, please determine if the newly recommended LIST is completely satisfactory.
'''

# build memory
rec_build_memory='''In round {}, the LIST of businesses you recommended was {}.
The reason you gave for the recommendation was: {}
The reason the user provided for rejecting this list was: {}
'''
user_build_memory='''In round {}, the list of recommended businesses was {}.
The reason given by the recommendation system was: {}
The reason you provided for not considering this the best recommendation was: {}
'''
user_build_memory_2='''In round {}, the list of recommended businesses was {}.
The reason given by the recommendation system was: {}
'''