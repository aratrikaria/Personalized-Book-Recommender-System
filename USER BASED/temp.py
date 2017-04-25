import pandas as pd
import math, time
import numpy as np
import csv
from itertools import groupby
import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.cross_validation import KFold
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import math
import warnings
import sys
from math import sqrt

start = time.time()

# users =  8026324
# items =  2330066
# count =  22507155

filename = 'ratings_Books.csv'
f = open(filename,"r")

user_id = dict()
users = 0

item_id = dict()
items = 0

count = 0




for row in f:
    r = row.split(',')  
    if (user_id.has_key(str(r[0])) == False):
        user_id[str(r[0])] = users
        users += 1

    if (item_id.has_key(str(r[1])) == False):
        item_id[str(r[1])] = items
        items += 1

    count += 1
    if (count == 10):
        break

print 'users = ', users
print 'items = ', items
print 'count = ', count

a=np.arange(-10,10)
b = np.where(a>5)[0][0]
print b




def origDatainTuples(fileName):

    f = open(filename,"r")

    data = []

    count = 0

    for row in f:
        count += 1
        r = row.split(',')
        e = [user_id.get(r[0]), item_id.get(r[1]), int(float(r[2]))]
        data.append(e)
        if (count == 10):
            break
    
    f.close()
    
    return data



def reducedMatrix(data):

    eightyMat = np.zeros((users,items))
    twentyMat = np.zeros((users,items))
    for e in data:
        eightyMat[e[0]-1][e[1]-1] = e[2]

    for user in range(users):
        a = eightyMat[user]
        if np.count_nonzero(a) > 1: #numberOfRatedItems > 1
            firstRatedItem = np.where(a>0)[0][0] 
            twentyMat[user][firstRatedItem] = eightyMat[user][firstRatedItem]
            eightyMat[user][firstRatedItem] = 0 #make the firstRatedItem = 0

    return eightyMat, twentyMat


def similarity_user(data):
    print "Hello User"
    user_similarity_jaccard = np.zeros((users,users))
    for user1 in range(users):
        print user1
        for user2 in range(users):
            if np.count_nonzero(data[user1]) and np.count_nonzero(data[user2]):
                user_similarity_jaccard[user1][user2] = 1-scipy.spatial.distance.jaccard(data[user1],data[user2])
    return user_similarity_jaccard


def calculateRMSE(data, eightyData, twentyData):

    sim_user_jaccard = sim_user

    rmse_jaccard = []
    mae_jaccard = []
    true_rate = []
    pred_rate_jaccard = []
    
    for e in twentyData:
        user = e[0]
        item = e[1]
        true_rate.append(e[2])
        pred_jaccard = 3.0

        #user-based
        if np.count_nonzero(eightyData[user-1]):
            sim_jaccard = sim_user_jaccard[user-1]
            ind = (eightyData[:,item-1] > 0)
            normal_jaccard = np.sum(np.absolute(sim_jaccard[ind]))

            if normal_jaccard > 0:
                pred_jaccard = np.dot(sim_jaccard,eightyData[:,item-1])/normal_jaccard

        if pred_jaccard < 0:
            pred_jaccard = 0

        if pred_jaccard > 5:
            pred_jaccard = 5

        print 'line 1:', str(user) + "\t" + str(item) + "\t" + str(e[2]) + "\t" + str(pred_jaccard)
        pred_rate_jaccard.append(pred_jaccard)


    ####################################   RMSE   ############################################ 
    rmse_jaccard.append(sqrt(mean_squared_error(true_rate, pred_rate_jaccard)))
    rmse_jaccard = sum(rmse_jaccard) / float(len(rmse_jaccard))

    ####################################   MAE  ############################################
    mae_jaccard.append(mean_absolute_error(true_rate, pred_rate_jaccard))
    mae_jaccard = sum(mae_jaccard) / float(len(mae_jaccard))

    print 'mae:', str(mae_jaccard)
    print 'rmse:', str(rmse_jaccard)

    #######################################################################
   




def user_recommendations(user, data, eightyData, twentyData):

    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list =[]

    for other in range(users):
        if other == user:
            continue
        sim = sim_user[user][other]

        # ignore scores of zero or lower
        if sim <=0: 
            continue
        for item in range(items):
            # only score movies i haven't seen yet
            if eightyData[other][item] == 0:
            # Similrity * score
                totals.setdefault(item,0)
                totals[item] += eightyData[other][item] * sim
                # sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+= sim

    # Create the normalized list
    rankings = [(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendataions_list = [recommend_item for score,recommend_item in rankings[0:10]]
    full_recommendataions_list = [recommend_item for score,recommend_item in rankings]
    return recommendataions_list, full_recommendataions_list



data = origDatainTuples(filename)
eightyData, twentyData = reducedMatrix(data)
sim_user = similarity_user(data)
# print eightyData
# print twentyData
# print 'sim_user: ', sim_user

#########################################
######Precision and Recall###############
#########################################
precision = 0.0
recall = 0.0
totalRecall = 0.0
for user in range(users):
    recList, fullRecList = user_recommendations(user, data, eightyData, twentyData)
    # print 'recList: ',recList
    # print 'fullRecList: ',fullRecList
    for rec in recList:
        if twentyData[user][rec] > 0:
            precision += 1.0

#######Recall#############################################################################
    recall = 0.0
    if(np.count_nonzero(twentyData[user]) > 0):
        for rec in fullRecList:
            if rec in twentyData[user]:
                recall += 1.0

        recall /=float(np.count_nonzero(twentyData[user]))

    totalRecall += recall

################################################F measure##################################
Fmeasure = 0.0
if precision * recall > 0:
    Fmeasure = float(2*precision*totalRecall)/((precision+totalRecall))

################################# RMSE AND MAE  ############################################
calculateRMSE(data, eightyData, twentyData)





print 'precision = ', precision
print 'totalRecall = ', totalRecall
print 'F measure = ', Fmeasure


print '\ntime taken : ', time.time()-start