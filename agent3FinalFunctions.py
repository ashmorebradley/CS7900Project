# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 22:39:35 2022

@author: Asus
"""
import tweepy
import collections
import datetime
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import haversine as hs
from haversine import Unit
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
from threading import Event
import sys
    
def UserAnalyzer(df, username, startDate, endDate, hashtag):
    if(startDate == "" and endDate == "" and hashtag == ""):
        print("\nTweets filtered by User " + username +": \n")
        tweetList = df['tweet'].values.tolist()
    elif(startDate != "" and endDate != "" and hashtag == ""):
        print("\nTweets filtered by User "+ username + " and Daterange " + startDate + "to " + endDate + ": " + " \n")
        df = df[(df['created_at'] > startDate) & (df['created_at'] < endDate)]
        tweetList = df['tweet'].values.tolist()
    elif(startDate == "" and endDate == "" and hashtag != ""):
        print("\nTweets filtered by User " + username + "and Hashtag: " + hashtag + "\n")
        df = df[(df['hashtags']) == hashtag]
        tweetList = df['tweet'].values.tolist()
    elif(startDate != "" and endDate != "" and hashtag != ""):
        print("\nTweets filtered by All Filters (User,Time,Hashtag:\n")
        df = df[(df['created_at'] > startDate) & (df['created_at'] < endDate) & (df['hashtags']) == hashtag]
        tweetList = df['tweet'].values.tolist()
        
    if(tweetList==[]):
        print("Empty Vocabulary Error")
        sys.exit()

    count_vect = CountVectorizer(ngram_range=(2,2), max_df = 0.8, stop_words='english', min_df=1)
    csr_mat = count_vect.fit_transform(tweetList)
    #print(csr_mat)

    features = count_vect.get_feature_names_out()
    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(csr_mat)

    dimension = 3
    lda = LDA(n_components = dimension)

    
    lda_array = lda.fit_transform(x_tfidf)

    
    print("\nTopic Probabilities In Texts: ", lda_array)
    sentiment = SentimentIntensityAnalyzer()
    lister = []
    print("\n")  
    print("Sentiment_Scores By User : " + username + "\n")
    for x in tweetList:
        polarity = sentiment.polarity_scores(x)
        #print(polarity)
        for x in polarity:
            lister.append(x)
    
    print(Counter(lister))      
    tweet_created_at = df['time']

    
    newTimeList = []
    for timeCr in tweet_created_at:
        timeCr = timeCr[:-3]
        newTimeList.append(timeCr)
    
    generalTimePosted = Counter(newTimeList)
     
    for y in generalTimePosted.copy():
        if(generalTimePosted[y] < 1):
            del generalTimePosted[y]
    
    
    '''plt.figure(figsize=(16,8))
    plt.bar(generalTimePosted.keys(), generalTimePosted.values())  '''
    
    hashtags = df['hashtags'].values
    
    hashtagList = []
    for x in hashtags:
        hashtagList.append(x)
    newListHashtag = []
    #print(hashtagList)
    for x in hashtagList:
        x = str(x)
        x = x.replace("[","")
        x = x.replace("]","")
        x = x.replace("'","")
        newListHashtag.append(x)
    
    mostUsedHashtags = Counter(newListHashtag)
     
    for y in mostUsedHashtags.copy():
        if((mostUsedHashtags[y] < 1) | (mostUsedHashtags[y] == 0)):
            del mostUsedHashtags[y]
    
    print("\n")
    print("Most Used Hashtags By the User " + username + "\n")
    print(mostUsedHashtags)
    '''plt.figure(figsize=(16,8))
    plt.bar(mostUsedHashtags.keys(), mostUsedHashtags.values())'''
    

    UserGroupAnalyzer(df, newListHashtag)
       
    
def UserGroupAnalyzer(df, hashtag):
    print("------- Topic Modelling for Users In a Group -------")
    
    tweet_geo = df['geo']
    #print(tweet_hashtags)
    tweet_geo = tweet_geo.to_list()

    #print(tweet_geo)
    user_hashtags = hashtag
    
    UserTweets = []
    groupUserslat = []
    groupUserslon = []
    lat_lon = (tweet_geo[0].split(','))
    lat1 = float(lat_lon[0])
    lon1 = float(lat_lon[1])
    dist = lat_lon[2].replace('km','')
    df = pd.read_csv('MasterDataLoc.csv')
    
    tweet_geo_all = df['geo']
    tweet_geo_all = tweet_geo_all.to_list()
    
    tweetText = df['tweet']
    tweet_Text = tweetText.to_list()
    
    tweet_hashtags = df['hashtags']
    tweet_hashtags = tweet_hashtags.to_list()
    
    for d in tweet_geo_all:
        d = d.split(',')
        lat2 = float(d[0])
        lon2 = float(d[1])
        loc1 = (lat1,lon1)
        loc2 = (lat2,lon2)
        distance = hs.haversine(loc1,loc2,unit=Unit.METERS)
        distance = (distance/1000)
        if(distance < float(dist)):
            groupUserslat.append(lat2)
            groupUserslon.append(lon2)
    
    for x in range(0,len(tweet_hashtags)):
        for y in tweet_hashtags[x]:
            if ((y in user_hashtags) & (tweet_geo_all[x] == ("25.7617,-80.1918,50km"))):
                UserTweets.append(tweet_Text[x])   
    
    count_vect = CountVectorizer(ngram_range=(2,2), max_df = 0.8, stop_words='english', min_df=1)
    csr_mat = count_vect.fit_transform(tweet_Text)
    #print(csr_mat)

    #features = count_vect.get_feature_names_out()
    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(csr_mat)

    dimension = 4
    lda = LDA(n_components = dimension, max_iter = 15)

    lda_array = lda.fit_transform(x_tfidf)
    
    '''for j in range(0,len(lda.components_[0])):
        features[j] = features[j] + ': t1= ' + str(lda.components_[0][j]) + ', ' + 't2=' + str(lda.components_[1][j])'''
       
    #print("Word-Topic Scores:", features)
    print("\nTopic Probabilities In Texts for the user group: ", lda_array)

    sentiment = SentimentIntensityAnalyzer()
    
    print("\n") 
    print("Sentiment_Scores for the user group \n: ")
    lister = []
    for x in tweet_Text:
        polarity = sentiment.polarity_scores(x)
        #print(polarity)
        for x in polarity:
            lister.append(x)
    
    print(Counter(lister))
            
    tweet_created_at = df['time']

    
    newTimeList = []
    for timeCr in tweet_created_at:
        timeCr = timeCr[:-3]
        newTimeList.append(timeCr)
    
    generalTimePosted = Counter(newTimeList)
     
    for y in generalTimePosted.copy():
        if(generalTimePosted[y] < 1):
            del generalTimePosted[y]
    
    
    '''plt.figure(figsize=(16,8))
    plt.bar(generalTimePosted.keys(), generalTimePosted.values())'''
    
    hashtags = df['hashtags'].values
    
    hashtagList = []
    for x in hashtags:
        hashtagList.append(x)
    newListHashtag = []
    #print(hashtagList)
    for x in hashtagList:
        x = str(x)
        x = x.replace("[","")
        x = x.replace("]","")
        x = x.replace("'","")
        newListHashtag.append(x)
    
    mostUsedHashtags = Counter(newListHashtag)
     
    for y in mostUsedHashtags.copy():
        if((mostUsedHashtags[y] < 1) | (mostUsedHashtags[y] == 0)):
            del mostUsedHashtags[y]
            
    print("Most used hashtags by group: ", mostUsedHashtags)
        

def main():
    df = pd.read_csv('MasterDataLoc.csv')
    userList = df['username']

    userList = userList.values.tolist()

    userList = (Counter(userList))

    mostOccuredUser = userList.most_common(1)

    for key,value in mostOccuredUser:
        username = key

    df = df[(df['username']==username)]
    htList = df['hashtags'].values.tolist()

    mostOccuredHashtag = (Counter(htList))
    mostOccuredHashtag = mostOccuredHashtag.most_common(3)

    userHashtag = "['democracy']"

    UserAnalyzer(df, username, "", "", "") 

    startDate =  '2022-03-27'
    endDate = '2022-03-30'

    UserAnalyzer(df, username, startDate, endDate, "") 

    UserAnalyzer(df, username, "", "", userHashtag)


if __name__=="__main__":
    main()






        
        

