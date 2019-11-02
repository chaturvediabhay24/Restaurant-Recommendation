#make necesarry imports
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine
# import ipywidgets as widgets
# from IPython.display import display, clear_output
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys, os
from contextlib import contextmanager

import pandas as pd
import numpy as np

rating_frame = pd.read_csv("Dataset/rating_final.csv")

for i in range(len(rating_frame['rating'])): # shifting rating such that 0 can be assigned to unreviewed items
  (rating_frame['rating'][i])+=1

restaurant_crosstab = pd.pivot_table(data=rating_frame, index='userID', columns='placeID', values='rating').fillna(0)

col_names=list(restaurant_crosstab)

def userID_to_int(ids):
  return int(ids[-3:])

#This function finds k similar users given the user_id and ratings matrix M
#Note that the similarities are same as obtained via using pairwise_distances
def findksimilarusers(user_id, metric = "cosine", k=4):
    ratings=restaurant_crosstab
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
    # print(distances, indices)
    similarities = 1-distances.flatten()
    # print ('{0} most similar users for User {1}:\n'.format(k,user_id))
    # for i in range(0, len(indices.flatten())):
    #     if indices.flatten()[i]+1 == user_id:
    #         continue;

    #     else:
    #         print ('{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i]+1, similarities.flatten()[i]))
            
    return similarities,indices


#This function predicts rating for specified user-item combination based on user-based approach
def predict_userbased(user_id, item_id, metric = "cosine", k=5):
    ratings=restaurant_crosstab
    prediction=0
    similarities, indices=findksimilarusers(user_id,metric, k) #similar users based on cosine similarity
    mean=ratings.mean(axis = 1, skipna = True) 
    mean_rating = mean[user_id-1] #to adjust for zero based indexing
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0 
    # print(similarities, indices)
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue;
        else: 
            ratings_diff = ratings.iloc[indices.flatten()[i],item_id-1]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product
    
    prediction = float((mean_rating + (wtd_sum/sum_wt)))
    # print ('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))

    return prediction

restraunt_cuisine = pd.read_csv("Dataset/chefmozcuisine.csv")
cuisine={}
for i, j in ((restraunt_cuisine.iterrows())):
  cuisine[str(j['placeID'])]=str(j['Rcuisine'])

#This function predicts rating for specified user-item combination based on user-based approach
def predict_items(user_id, metric = "cosine", count=5):
  ratings=restaurant_crosstab
  scores=[]
  cuisines=set()
  for i in range(1,131):
    score=[]
    if(int(ratings.iloc[user_id-1, i-1])>0):   # if already rated by user then dont need to predict it
      score.append(ratings.iloc[user_id-1, i-1])
      score.append(i)
      scores.append(score)
      continue;
    score.append(predict_userbased(user_id,i))
    score.append(i)
    scores.append(score)
  scores.sort(key=lambda x:x[0], reverse=True)
  for i in scores:
    ids=col_names[i[1]-1]
    # print(ids)
    if str(ids) in cuisine.keys():
      cuisine_name=cuisine[str(ids)]
      cuisines.add(cuisine_name)
    if(len(cuisines)>=count):
      break
    
  return (cuisines)

restraunt_cuisine = pd.read_csv("Dataset/userprofile.csv")
activity={}
for i, j in ((restraunt_cuisine.iterrows())):
  activity[str(j['userID'])]=str(j['activity'])

def give_activity(userid):
	return activity[str(userid)];

restraunt_cuisine = pd.read_csv("Dataset/userpayment.csv")
upayment={}
for i, j in ((restraunt_cuisine.iterrows())):
  upayment[str(j['userID'])]=str(j['Upayment'])

def give_upayment(userid):
	return upayment[str(userid)];