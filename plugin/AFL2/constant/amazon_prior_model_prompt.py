
rec_system_prompt='''You are a product recommendation system.
Refine the user's shopping history to predict the Top 5 most likely products they will purchase next from a selection of candidates.
Another recommendation system has provided its recommended product, which you can refer to.

Some useful tips: 
1. You need to first give the reasons, and then provide the recommended products.
2. The products you recommend must be in the candidate list.
3. List them in order from most likely to least likely.

You must follow this output format: 
Reason: <your reason example>
Items: <item1>, <item2>, <item3>, <item4>, <item5> 
'''

rec_user_prompt='''This user has purchased {} in the previous.
Given the following {} products: {}, you should recommend the Top 5 products for this user to purchase next.
The product recommended by another recommendation system is: {}.
Based on the above information, you can select the product recommended by another recommendation system as part of your list or choose other products from the candidates.
'''

rec_memory_system_prompt='''You are a product recommendation system.
Refine the user's shopping history to predict the Top 5 most likely products they will purchase next from a selection of candidates.
However, the user previously felt that the products you recommended were not their top choices.
Based on the feedback and history, select the best Top 5 products again from the candidate list.

Some useful tips: 
1. You need to first give the reasons, and then provide the recommended products.
2. The products you recommend must be in the candidate list.
3. Rank them from 1 to 5 (most likely to least likely).

You must follow this output format: 
Reason: <your reason example>
Items: <item1>, <item2>, <item3>, <item4>, <item5> 
'''

rec_memory_user_prompt='''This user has purchased {} in the previous.
Given the following {} products: {}, you should recommend Top 5 products for this user to purchase next.
Here are the lists of products you previously recommended and the reasons why the user thinks they are not the best choices:
{}

Based on the above information, select a new set of Top 5 products again from the candidate list.
'''


user_system_prompt='''As a shopper, you've purchased the following products: {}.
The specific product you are actually looking for (your target) is: {}.
Now, a recommendation system has recommended a LIST of Top 5 products to you from a list of candidates, and has provided the reasons for the recommendation.
Determine if this recommended LIST **is completely satisfactory**, meaning it is highly likely to contain the target product you would purchase next, or if you need the system to try again.

Some useful tips:
1. You need to first give the reasons, and then decide whether or not the recommended LIST is satisfactory.
2. Use "yes" to indicate that the LIST is satisfactory (the target product is in the list), and use "no" to indicate that you want the system to try again.
3. Summarize your own interests based on your historical records to make a judgment.
4. You should only say "Yes" if you are highly confident that your target product is among the Top 5 recommended items.

You must follow this output format: 
Reason: <your reason example>
Decision: <yes or no>
'''

user_user_prompt='''The list of candidate products is: {}.
You can focus on considering these products: {}.
The LIST of RECOMMENDED products (Top 5) is: {}.
The reason provided by the recommendation system for this list is: {}
Please determine if this recommended LIST is completely satisfactory.
'''

user_memory_system_prompt='''As a shopper, you've purchased the following products: {}.
The specific product you are actually looking for (your target) is: {}.
Previously, a recommendation system attempted to select your favorite products from a list of candidates, but you were not satisfied with those lists.
Now, the recommendation system has once again recommended a new LIST of Top 5 products and provided its reasons.
Determine if this new recommended LIST **is completely satisfactory** (contains your target product), or if you need the system to try again.

Some useful tips:
1. You need to first give the reasons, and then decide whether or not the recommended LIST is satisfactory.
2. Only use "yes" to indicate that the LIST is satisfactory, and "no" to indicate that you want the system to try again.
3. Summarize your own interests based on your historical records AND the communication history to make a judgment.
4. You should only say "Yes" if you feel highly confident that your target product is in this new list.

You must follow this output format: 
Reason: <your reason example>
Decision: <yes or no>
'''

user_memory_user_prompt='''The list of candidate products is: {}.
You can focus on considering these products: {}.
Here are the lists of products previously recommended by the recommendation system and your reasons for rejecting them:
{}

Now, the new LIST of RECOMMENDED products (Top 5) is: {}.
The recommendation system provides the following reason for this list: {}
Based on the above information, please determine if the newly recommended LIST is completely satisfactory.
'''


rec_build_memory='''In round {}, the LIST of products you recommended was {}.
The reason you gave was: {}
The reason the user provided for rejecting this list was: {}
'''

user_build_memory='''In round {}, the list of recommended products was {}.
The reason given by the recommendation system was: {}
The reason you provided for not considering this the best recommendation was: {}
'''

user_build_memory_2='''In round {}, the list of recommended products was {}.
The reason given by the recommendation system was: {}
'''