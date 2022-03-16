# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:17:38 2022

@author: ashmo
"""


#########################################
#TopicRecoverer Agent
#########################################


#########################################
#Variables used by the agent
#########################################
topicDataRaw = {}
#topicData  is a topic-indexed dictionary. Each topic points to a list of tuples containing content, user, time, and a list of hashtags)

topicDataProcessed = {}
#topicData is a topic-indexed dictionary. Each topic name points to a list of statistics that can be used to decide the most-likely topic of a deleted tweet.
#Contents of this diciton ary are TBD.


#########################################
#Function used to add data to the agent.
#User - A string.
#Content - A string
#Hashtag - A string
#locaiton - A tuple of lat/lon values, floats
#time - A string or date time object
#topic - A string


def loadData(user = 0, content = 0, hashtag = 0, location = 0, time = 0, topic = 0):
    #This funciton loads data into the topicDataRaw dictionary
    print("Done")
    
def processRawData():
    #This funciton process data in the topicDataRaw variable into a usable form and stores the result in topicDataProcessed
    
    
    print("Done")
    
    
#This fuction uses data in the topicDataProcessed variable to determine the most-likely topic.    
def determineTopic( User, hashtags, location, time):
    
    print("Done")