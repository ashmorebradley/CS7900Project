# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:17:38 2022

@author: ashmo
"""


#########################################
#TopicRecoverer Agent
#########################################
import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import TopicAssigner #TopicRecoverer
import json

#########################################
#Variables used by the agent
#########################################
topicDataRaw = {}
#topicData  is a topic-indexed dictionary. Each topic points to a list of tuples containing content, user, time, and a list of hashtags)

topicDataProcessed = {}
#topicData is a topic-indexed dictionary. Each topic name points to a list of statistics that can be used to decide the most-likely topic of a deleted tweet.
#Contents of this diciton ary are TBD.

knownTopics = []

probMatrix = {}
probMatrixUser = {}
probMatrixHash = {}
probMatrixLoc = {}


stopWords = text.ENGLISH_STOP_WORDS.union({"http", "https", "Https", "Https:", "t", "s"})

#########################################
#Function used to add data to the agent.
#User - A string.
#Content - A string
#Hashtag - A string
#locaiton - A tuple of lat/lon values, floats
#time - A string or date time object
#date - A string or date time object
#topic - A string

def removeStopwords(l, stops):
    toReturn = []
    for word in l:
        for s in stops:
            if(word != s):
                toReturn.append(word)
                
    return toReturn
########################################
##Accepts user data and reuturns a list of topics
##Returns a dictionary of topic : prob.
##A simple baysian analysis
####################################
def decodeListOfTopics(userD):
    toReturn = {}
    for topic in userD:
        if topic in toReturn:
            toReturn[topic] += 1
        else:
            toReturn[topic] = 1
            
    numTopics = len(userD)
    
    for key in toReturn.keys():
        toReturn[key] = toReturn[key] / numTopics
        
    toReturn["knownTopicCount"] = numTopics

#####################
    #Updates values when a new topic is seen
def updateTopicsDictionary(topDic):
    divisor = topDic["knownTopicCount"]
    toReturn = topDic
    for key in toReturn.keys():
        if key != "knownTopicCount":
            toReturn[key] = (toReturn[key] + 1) / divisor
            
    return toReturn
            
        
      
    
#Content shouldn't be given.
def loadData(tweetId, content = 0, user = 0, hashtag = 0, location = 0, date = 0, time = 0, topic = 0):
    #Get known topics.
    knownTopics = PartialTopicAssigner.getKnownTopics()
    
    tweets = []
    #This is fragile. It assumes that all input arrays are the same length.
    #This should always be true but if something goes wrong there is no check.
    for i in range(len(tweetId)):
        tweets.append( (tweetId[i], content[i], user[i], hashtag[i], location[i], date[i], time[i]  ) )
        
    userData = {} #In the form User : List of topics
    hashTagData = {}
    locationData = {}
    dateData = {}
    
    for twt in tweets:
        #3 is user
        topicHold = PartialTopicAssigner.determineTopic(twt[1])
        print("TOPIC IS::      " + str(topicHold))
        #Do user things
        if twt[2] in userData:
            userData[twt[2]].append( topicHold )
        else:
            userData[twt[2]] = [ topicHold ]
        
        #Do user things
        if twt[3] != '[]':      #Ignore empty hashtags.
            lst = twt[3][1:-1]
            lst = lst.split(',')
            for i in range(len(lst)):
                lst[i] = lst[i][1:-1]
                
      #      lst = json.loads(twt[3])    #Convert from JSON string to python list.
            for ht in lst:
                if ht in hashTagData:
                    hashTagData[ht].append( topicHold )
                else:
                    hashTagData[ht] = [ topicHold ]
                
            
        #Do location things
        if twt[4] in locationData:
            locationData[twt[4]].append( topicHold )
        else:
            locationData[twt[4]] = [ topicHold ]
        
        #Do date things
        if twt[5] in dateData:
            dateData[twt[5]].append( topicHold )
        else:
            dateData[twt[5]] = [ topicHold ]
            

    #Manipulate topic lists...
    #For every user, calculate the topic frequency.
    for key in userData.keys():
        topicDic = {}
        topics = userData[key]
        #For every topic entry...
        for topic in topics:
            if topic in topicDic:
                topicDic[topic] = topicDic[topic] + 1
            else:
                topicDic[topic] = 1
                
        #Have all the topic totals, now convert to a probability.
        for topic in topics:
            topicDic[topic] = topicDic[topic] / len(topics)
        
        #Got the dictonary of topics statistics....Add to the mater dictionary. 
        probMatrixUser[key] = topicDic
    
    #For every hashtaag
    for key in hashTagData:
       topicDic = {}
       topics = hashTagData[key]
       #For every topic entry...
       for topic in topics:
           if topic in topicDic:
               topicDic[topic] = topicDic[topic] + 1
           else:
               topicDic[topic] = 1
                
        #Have all the topic totals, now convert to a probability.
       for topic in topics:
           topicDic[topic] = topicDic[topic] / len(topics)
        
        #Got the dictonary of topics statistics....Add to the mater dictionary. 
       probMatrixHash[key] = topicDic
        
    #For every location.  
    for key in locationData:
       topicDic = {}
       topics = locationData[key]
       #For every topic entry...
       for topic in topics:
           if topic in topicDic:
              topicDic[topic] = topicDic[topic] + 1
           else:
              topicDic[topic] = 1
                
        #Have all the topic totals, now convert to a probability.
       for topic in topics:
           topicDic[topic] = topicDic[topic] / len(topics)
        
        #Got the dictonary of topics statistics....Add to the mater dictionary. 
       probMatrixLoc[key] = topicDic
    

    
#This fuction uses data in the topicDataProcessed variable to determine the most-likely topic.        
def determinTopic(user = None, hashtags = None, location = None, date = None, time = None):
    #Calculate probabilities
    #P(topic | User)
    userProbs = []
    hashProbs = []
    locProbs = []
    
    if user != None:
        userProbs = probMatrixUser[user]
        
    #P(topic | Hashtag)
    if hashtags != None: 
       hashProbs = probMatrixHash[hashtags]
    
    #P(Topic | Location)
    if location != None:
        locProbs = probMatrixLoc[loc]
    
    maxLen = max(len(userProbs), len(hashProbs), len(locProbs))
    inuse = ( len(userProbs) > 0, len(hashProbs) > 0, len(locProbs) > 0 )
    
    finalProbs = []
    for i in range(maxLen):
        if inuse[0] and inuse[1] and inuse[2]:
            finalProbs.append(userProbs[i][1] * hashProbs[i][1] * locProbs[i][1])
            
        elif inuse[0] and inuse[1]:
            finalProbs.append(userProbs[i][1] * hashProbs[i][1] )    
            
        elif inuse[0] and inuse[2]:
            finalProbs.append(userProbs[i][1]  * locProbs[i][1])
            
        elif inuse[1] and inuse[2]:
            finalProbs.append(hashProbs[i][1] * locProbs[i][1])
            
        elif inuse[0]:
            finalProbs.append(userProbs[i][1])
            
        elif inuse[1]:
            finalProbs.append(hashProbs[i][1] )
        
        elif inuse[2]:
            finalProbs.append(locProbs[i][1])
        
    #Find max
    mx = 0 
    loc = 0
    i = 0
    for x in finalProbs:
        if x > mx:
            mx = x
            loc = i
        
        i += 1
        
    return learnedTopics[i]