# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 22:39:35 2022

@author: Asus
"""
import tweepy
import collections
import datetime
from datetime import datetime
consumer_key = "XXXXXXXXXXX"
consumer_secret = "XXXXXXXXXXXX"
access_key = "XXXXXXXXXXXXXX"
access_secret = "XXXXXXXXXXX"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

def FindHashtags(collected_tweets):
    hashtagsUsed = []
    for i in range(0,len(collected_tweets)):
        if(collected_tweets[i]['entities']['hashtags']==[]):
           continue
        hashtagsUsed.append(collected_tweets[i]['entities']['hashtags'][0]['text'])
    print(collections.Counter(hashtagsUsed))
     
def FindTimePosted(collected_tweets):
    timesPosted = []
    for i in range(0,len(collected_tweets)):
        timePosted = collected_tweets[i]['created_at'].replace('+0000','')
        timePosted = timePosted.replace('  ',' ')
        timePosted = datetime.strptime(timePosted, "%a %b %d %H:%M:%S %Y")
        timesPosted.append(timePosted.time())
    print(timesPosted)


new_tweets = tweepy.Cursor(api.user_timeline, screen_name="leemickus", tweet_mode='extended').items(5)

collected_tweets = []
for tweet in new_tweets:
    collected_tweets.append(tweet._json)

FindHashtags(collected_tweets)
FindTimePosted(collected_tweets)



        
        

