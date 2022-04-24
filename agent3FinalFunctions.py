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

consumer_key = "XXXXXXXXXX"
consumer_secret = "XXXXXXXXXX"
access_key = "XXXXXXXXXX"
access_secret = "XXXXXXXXXX"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


def FindTopics():
    newTweets = []
    df = pd.read_csv('MasterDataLoc.csv')
    #a = (df.query('username=="conrad1985"')['tweet'])
    userList =  df.username.values
    print(Counter(userList))
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
    
    print("Topic Probabilities In Texts: ", lda_array)
    sentiment = SentimentIntensityAnalyzer()
    
        
    for x in tweetList:
        print("Sentiment_Scores: " ,sentiment.polarity_scores(x))
    
def FindTopicsInTime():
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
    
    print("Topic Probabilities In Texts: ", lda_array)
    sentiment = SentimentIntensityAnalyzer()
    
    for x in tweetList:
        print("Sentiment_Scores: " ,sentiment.polarity_scores(x))
        
def FindTopicsGroup():
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
    print("Topic Probabilities In Texts: ", lda_array)
    print(lda_array)
        
def FindHashtags():
    new_tweets = tweepy.Cursor(api.user_timeline, screen_name="libertyandfree4", tweet_mode='extended').items(500)

    collected_tweets = []
    
    df = pd.read_csv('MasterDataLoc.csv')
    #a = (df.query('username=="conrad1985"')['tweet'])
    userList =  df.username.values
    print(Counter(userList))
    
    for tweet in new_tweets:
        collected_tweets.append(tweet._json)
    hashtagsUsed = []
    mostUsedHashtags = []
    CounterFormostUsedHashtags = []
    for i in range(0,len(collected_tweets)):
        if(collected_tweets[i]['entities']['hashtags']==[]):
           continue
        else:
           hashtagsUsed.append(collected_tweets[i]['entities']['hashtags'][0]['text'])
    CalcMostUsed = Counter(hashtagsUsed)
    for x in CalcMostUsed.copy():
        if(CalcMostUsed[x] < 2):
            del CalcMostUsed[x]
    print(CalcMostUsed)
    
    plt.figure(figsize=(16,8))
    plt.bar(CalcMostUsed.keys(), CalcMostUsed.values())
    
def FindHashtagsInTime():
    '''startDate = datetime.strptime('2021-03-30 13:00:00+0000', "%Y-%m-%d %H:%M:%S%z")
    endDate = datetime.strptime('2020-05-01 13:00:00+0000', "%Y-%m-%d %H:%M:%S%z")'''
    newTweets = []
    df = pd.read_csv('MasterDataLoc.csv')
    
    tweet_created_at = df['created_at']
    tweet_username = df['username']
    tweet_hashtags = df['hashtags']

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
    new_tweets = tweepy.Cursor(api.user_timeline, screen_name="libertyandfree4", tweet_mode='extended').items(50)

    collected_tweets = []
    timesPosted = []
    for tweet in new_tweets:
        collected_tweets.append(tweet._json)
    for i in range(0,len(collected_tweets)):
        timePosted = collected_tweets[i]['created_at'].replace('+0000','')
        timePosted = timePosted.replace('  ',' ')
        timePosted = datetime.strptime(timePosted, "%a %b %d %H:%M:%S %Y")
        timesPosted.append(timePosted.time())
    
    CalcMostUsed = Counter(timesPosted)
    for x in CalcMostUsed.copy():
        if(CalcMostUsed[x] < 1):
            del CalcMostUsed[x]
    
    print(CalcMostUsed)


FindHashtags()
FindHashtagsInTime()
FindTimePosted()

FindTopics()
FindTopicsInTime()
    
FindTopicsGroup()


        
        

