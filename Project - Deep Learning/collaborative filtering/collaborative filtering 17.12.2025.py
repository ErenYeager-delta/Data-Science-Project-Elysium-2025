#collaborative filtering
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Small user–item rating table
ratings = pd.DataFrame({
    'User':  ['A',  'B',  'C'],
    'Item1': [5,    4,    0],
    'Item2': [3,    0,    4],
    'Item3': [0,    2,    5],
}).set_index('User')

# 2. Find how similar users are
user_similarity = cosine_similarity(ratings)
user_similarity = pd.DataFrame(user_similarity,
                               index=ratings.index,
                               columns=ratings.index)

# 3. Simple recommend function
def recommend(user):
    similar_users = user_similarity[user].sort_values(ascending=False)[1:]
    # takes one column (e.g., similarities with user A), sorts from most
    # similar to least, and removes A itself
    scores = {}#store predicted scores for items user A has not rated

    for other_user, sim in similar_users.items():
        for item in ratings.columns:#Check every item
            # only items the target user has not rated
            if ratings.loc[user, item] == 0 and ratings.loc[other_user, item] > 0:
              #Add similarity × rating to that item’s score (weighted sum).
                scores[item] = scores.get(item, 0) + sim * ratings.loc[other_user, item]

    # sort by score (high to low)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # NEW (only top 2)
    #top2 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    #return top2

# 4. Show recommendations for user A
print("Recommendations for A:")
for item, score in recommend('A'):
    print(item, "->", round(score, 2))
