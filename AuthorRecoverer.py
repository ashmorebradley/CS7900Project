# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:46:49 2022

@author: ashmo
"""

#########################################
#AuthorRecoverer Agent
#########################################
userDataRaw = {}
#userData  is a username-indexed dictionary. Each user name points to a list of tuples containing content, topic, time, and a list of hashtags)

userDataProcessed = {}
#userData  is a username-indexed dictionary. Each user name points to a list of statistics that can be used to decide what user created a deleted tweet.
#Contents of this diciton ary are TBD.

#########################################
#Variables used by the agent
#########################################

#########################################
#Function used to add data to the agent.
#User - A string.
#Content - A string
#Hashtag - A string
#locaiton - A tuple of lat/lon values, floats
#time - A string or date time object
#topic - A string


def loadData(user, content = 0, hashtag = 0, location = 0, time = 0, topic = 0):
    #This function loads data into the userData dictionary.
    #This function needs to ensure that it does not duplicate existing users.
 
    return "" #This will need to be removed once the function is complete.

def processRawData():
    #This funciton process data in the userDataRaw variable into a usable form and stores the result in userDataProcessed
    
    
    return "" #This will need to be removed once the function is complete.

def determineAuthor(Content, hashtag, location, time):
    #This fucntion utilizes the data in userDataProcessed and returns the most-likely author.
    author = ""
    
    return author