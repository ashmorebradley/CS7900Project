# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:37:43 2022

@author: ashmo
"""
#########################################
#UserAnalyzer Agent
#########################################

#########################################
#Variables used by the agent
#########################################
userHash = {} 
#UserHash is a username-indexed dictionary. Each user name points to a list of tuples containing topic, time, and a list of hashtags)
#{User ==> ([ [topic, time, [hashtag]] ] )}
#I invision each tweet to be its own entry in the list.

#########################################
#Function used to add data to the agent.
#User - A string.
#Topics - A list of strings.
#Time - ???? Probably a list of int of some kind?, this should proably reflect the date as well.
#Hashtags - A list of string.
def loadData(user, topics, time, hashtags):
    #This funciton should check for an existing user.
    #If the user is present add new data to the dictionary.
    #If the user is not present, create an entry for the user.
    
    return "" #This will need to be removed once the function is complete.

#User - A string
def commonTopic(user):
    #Looks up a user and returns the most common topic & the probability of that topic
    
    return(topic, prob)
    
def commonTopicAtTime(user, time)    :
    #Looks up a user and returns the most common topic at a particular time & the probability of that topic at the time.
    #This probably needs to be over a window of time
    
    return(topic, prob)
    
def frequentPostTimes(user):
    #Looks up a user and returns the most frequent post time of that user.
    
    return(time)

def commonHashtags(user):
    #Looks up a user and returns the most frequent hashtag of that user and probability.

    return (ht, prob)


def commonHashtags(user, time):
    #Looks up a user and returns the most common hashtag at a particular time & the probability of that hashtag at the time.

    return (ht, prob)
