# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:01:49 2022

@author: ashmo
"""

import re
import pandas as pd
import AuthorRecoverer
import TopicRecoverer

#Get header data
#cols = pd.read_csv('baseball_tweets.csv').columns
#masterFrame = pd.DataFrame(columns = cols)
data = pd.read_csv('tweets.csv')
print(list(data.columns))

#All columns not used. Included to simplify future additions, if any. 
#allColumns = ['Unnamed: 0', 'id', 'conversation_id', 'created_at', 'date', 'time', 'timezone', 'user_id', 'username', 'name', 'place', 'tweet', 'language', 'mentions', 'urls', 'photos', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url', 'video', 'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src', 'trans_dest']
columnsUsed = ['Unnamed: 0', 'id', 'conversation_id', 'created_at', 'date', 'time', 'timezone', 'user_id', 'username', 'name', 'place', 'tweet', 'language', 'replies_count', 'hashtags', 'link', 'retweet', 'geo']

data = data[columnsUsed]
#data.rename(columns = {'Unnamed: 0 ': 'tweet_id'})
#print(data["Unnamed: 0"])
##Remove time zone from created date.


TopicRecoverer.loadData(tweetId = data[' Unnamed: 0'], user = data["user_id"], content = data['tweet'], hashtag = data['hashtags'], location = data['geo'], date = data['date'], time = data['time'])  
#Topics need to established first.
#AuthorRecoverer.loadData(data["user_id"], data["tweet", data["hashtags"], data["geo"], data['date'], date['time'], data['topic'] ])
#def loadData(user, content = 0, hashtag = 0, location = 0, time = 0, topic = 0):