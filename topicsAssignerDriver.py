# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 01:08:15 2022

@author: ashmo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 12:01:49 2022

@author: ashmo
"""

import re
import pandas as pd
import PartialTopicAssigner #TopicRecoverer


data = pd.read_csv('MasterDataLoc.csv')

#All columns not used. Included to simplify future additions, if any. 
columnsUsed = ['Unnamed: 0', 'id', 'conversation_id', 'created_at', 'date', 'time', 'timezone', 'user_id', 'username', 'name', 'place', 'tweet', 'language', 'replies_count', 'hashtags', 'link', 'retweet', 'geo']

data = data[columnsUsed]

PartialTopicAssigner.loadData(tweetId = data['Unnamed: 0'], user = data["user_id"], content = data['tweet'], hashtag = data['hashtags'], location = data['geo'], date = data['date'], time = data['time'])  
print("=================================")
print(PartialTopicAssigner.getKnownTopics())
print("")
print(PartialTopicAssigner.topicsKeyWords)
print("\n=================================")

tweetNum = 70

print("TEST TWEET: \n" + data['tweet'][tweetNum] )
print(PartialTopicAssigner.determineTopic("Topic Determined: " + str(data['tweet'][tweetNum]) ))
print("Finished")