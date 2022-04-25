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

def FindTopics():
    print("------- Topic Modelling for User -------")
    newTweets = []
    df = pd.read_csv('MasterDataLoc.csv')
    #a = (df.query('username=="conrad1985"')['tweet'])
    userList =  df.username.values

    tweetList = df.loc[df.username=='conrad1985', 'tweet'].values
    #print(tweetList)
    count_vect = CountVectorizer(ngram_range=(2,2), max_df = 0.8, stop_words='english', min_df=1)
    csr_mat = count_vect.fit_transform(tweetList)
    #print(csr_mat)

    features = count_vect.get_feature_names_out()
    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(csr_mat)

    dimension = 2
    lda = LDA(n_components = dimension)

    
    lda_array = lda.fit_transform(x_tfidf)
    for j in range(0,len(lda.components_[0])):
        features[j] = features[j] + ': t1= ' + str(lda.components_[0][j]) + ', ' + 't2=' + str(lda.components_[1][j])
            
    print("Word-Topic Scores:", features)
    
    print("\nTopic Probabilities In Texts: ", lda_array)
    sentiment = SentimentIntensityAnalyzer()
    
    print("\n")  
    for x in tweetList:
        print("Sentiment_Scores: " ,sentiment.polarity_scores(x))
    
    
def FindTopicsInTime():
    print("------- Topic Modelling for User In a Time-Frame -------")
    newTweets = []
    df = pd.read_csv('MasterDataLoc.csv')
    
    df.created_at = df.created_at.str.replace("Eastern Daylight Time","")
    #df = df[(df['username']=='conrad1985')]
    df = df[(df['created_at'] > '2022-03-26') & (df['created_at'] < '2022-03-29') & (df['username']=='conrad1985')]
    #print(df['created_at'])
    #print(df)
    tweetList = df['tweet'].values
    #print(tweetList)
    
    count_vect = CountVectorizer(ngram_range=(2,2), max_df = 0.8, stop_words='english', min_df=1)
    csr_mat = count_vect.fit_transform(tweetList)
    features = count_vect.get_feature_names_out()
    #print(count_vect.get_feature_names_out())
    #print(csr_mat)


    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(csr_mat)

    dimension = 2
    lda = LDA(n_components = dimension)

    lda_array = lda.fit_transform(x_tfidf)
    for j in range(0,len(lda.components_[0])):
        features[j] = features[j] + ': t1= ' + str(lda.components_[0][j]) + ', ' + 't2=' + str(lda.components_[1][j])
            
    print("Word-Topic Scores:", features) 
    
    print("Word-Topic Scores:", features)
    
    print("\nTopic Probabilities In Texts: ", lda_array)
    sentiment = SentimentIntensityAnalyzer()
    
    print("\n")  
    for x in tweetList:
        print("Sentiment_Scores: " ,sentiment.polarity_scores(x))
    
    
def FindTopicsGroup():
    print("------- Topic Modelling for Users In a Group -------")
    newTweets = []
    df = pd.read_csv('MasterDataLoc.csv')
    
    tweet_created_at = df['created_at']
    tweet_username = df['username']
    tweet_hashtags = df['hashtags']
    tweet_geo = df['geo']
    tweetText = df['tweet']

    # convert from pandas to lists
    tweet_created_at = tweet_created_at.to_list()
    tweet_username = tweet_username.to_list()
    tweet_hashtags = tweet_hashtags.to_list()
    #print(tweet_hashtags)
    tweet_geo = tweet_geo.to_list()
    tweet_Text = tweetText.to_list()
    #print(tweet_geo)
    user_hashtags = ['amc','ukraine','iealfjoiea']
    user_lattitude = 41
    user_longitude = -74
    geo_array = np.array([user_lattitude, user_longitude])
    
    #print(tweet_geo)
    
    #print(tweet_geo[0].split(','))
    UserTweets = []
    groupUsers1 = []
    for d in tweet_geo:
        d = d.split(',')
        lat1 = float(d[0])
        lon1 = float(d[1])
        lat2 = geo_array[0]
        lon2 = geo_array[1]
        loc1 = (lat1,lon1)
        loc2 = (lat2,lon2)
        dist = hs.haversine(loc1,loc2,unit=Unit.METERS)
        dist = (dist/1000)
        if(dist < 50):
            groupUsers1.append(lon1)
    #print(groupUsers1)
    
    
    for x in range(0,len(tweet_hashtags)):
        for y in tweet_hashtags[x]:
            if (y in user_hashtags) & (tweet_geo[x] == '40.7128,-74.0060,50km'):
                UserTweets.append(tweet_Text[x])   
    
    count_vect = CountVectorizer(ngram_range=(2,2), max_df = 0.8, stop_words='english', min_df=1)
    csr_mat = count_vect.fit_transform(tweet_Text)
    #print(csr_mat)

    features = count_vect.get_feature_names_out()
    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(csr_mat)

    dimension = 4
    lda = LDA(n_components = dimension, max_iter = 15)

    lda_array = lda.fit_transform(x_tfidf)
    
    for j in range(0,len(lda.components_[0])):
        features[j] = features[j] + ': t1= ' + str(lda.components_[0][j]) + ', ' + 't2=' + str(lda.components_[1][j])
       
    print("Word-Topic Scores:", features)
    print("\nTopic Probabilities In Texts: ", lda_array)
    print(lda_array)
        
def FindHashtags():
    print("------ Most Frequent Hashtags Used By User -------")
    newTweets = []
    df = pd.read_csv('MasterDataLoc.csv')
    
    tweet_created_at = df['created_at']
    tweet_username = df['username']
    
    
    tweet_username = tweet_username.to_list()

    df = df[(df['username']=='conrad1985')]
 
    #tweet_hashtags = df['hashtags'].to_list()
    hashtags = df['hashtags'].values.flatten()
    
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
        #print(x)
        newListHashtag.append(x)
    
    mostUsedHashtags = Counter(newListHashtag)
     
    for y in mostUsedHashtags.copy():
        if(mostUsedHashtags[y] < 1):
            del mostUsedHashtags[y]
    
    
    plt.figure(figsize=(16,8))
    plt.bar(mostUsedHashtags.keys(), mostUsedHashtags.values())
    
    
def FindHashtagsInTime():
    print("------- Most Frequent Hashtags Used By User In a time frame: -------")
    newTweets = []
    df = pd.read_csv('MasterDataLoc.csv')
    
    tweet_created_at = df['created_at']
    tweet_username = df['username']
    tweet_hashtags = df['hashtags']
    
    tweet_created_at = tweet_created_at.to_list()
    tweet_username = tweet_username.to_list()
    tweet_hashtags = tweet_hashtags.to_list()

    df = df[(df['created_at'] > '2022-03-21') & (df['created_at'] < '2022-03-29') & (df['username']=='conrad1985')]
 
    hashtags = df['hashtags'].values.flatten()
    
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
        #print(x)
        newListHashtag.append(x)
    
    mostUsedHashtags = Counter(newListHashtag)
     
    for y in mostUsedHashtags.copy():
        if(mostUsedHashtags[y] < 1):
            del mostUsedHashtags[y]
    
    
    plt.figure(figsize=(16,8))
    plt.bar(mostUsedHashtags.keys(), mostUsedHashtags.values())
     
     
def FindTimePosted():
    print("------ Time of the Day User Posts with Frequency -------")
    newTweets = []
    df = pd.read_csv('tweets.csv')
    
    df = df[(df['username']=='swissdog3')]
    
    tweet_created_at = df['time']

    
    tweet_created_at = tweet_created_at.to_list()
    
    newTimeList = []
    for timeCr in tweet_created_at:
        timeCr = timeCr[:-3]
        newTimeList.append(timeCr)
    
    generalTimePosted = Counter(newTimeList)
     
    for y in generalTimePosted.copy():
        if(generalTimePosted[y] < 2):
            del generalTimePosted[y]
    
    
    plt.figure(figsize=(16,8))
    plt.bar(generalTimePosted.keys(), generalTimePosted.values())
    
    
    
    


FindHashtags()
#FindHashtagsInTime()


#FindTopics()

#FindTopicsInTime()

#FindTopicsGroup()

#FindTimePosted()
    



        
        

