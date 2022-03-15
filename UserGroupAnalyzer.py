# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 23:01:28 2022

@author: ashmo
"""
#########################################
#UserGroupAnalyzer Agent
#########################################

#########################################
#Variables used by the agent
#########################################
groupList = [] 
#Group is a list of users. My intention is that groups are id'ed by number 0 to n. Each entry contains a list of users. 
#These assumptions can change

ungroupedUsers {}
#ungroupUsers is a user-indexed dictionary. Each username points to a list of friends. 

#########################################
#Function used to add data to the agent.
#User - A string.
#Frieds - A list of string
def loadData(user, friends):

    
    
    #Retruns a list of lists?? 
def generateUserGroups([Users]):
    #This function should cluster users from the "ungropupedUsers" dictionary.
    #This function will need to ensure it does not duplicate pre-grouped users.
    #Assign result to groupList and empty ungroupsUsers
    return ""

#groupIdentifier - Integer associated with a group
def commonTopics(groupIdentifier):
    #Looks up a usergroup and returns the most common topic & the probability of that topic

    return (topic, prob)


def CommonTopicsAtTime(groupIdentifier, time):
    #Looks up a usergroup and returns the most common topic at a particular time & the probability of that topic

    return (topic, prob)

def CommonHashtags(groupIdentifier):
    #Looks up a usergroup and returns the most common hashtag & the probability of that tag
    
    return (ht, prob)

def CommonHashtags(GroupIdentifier, time):
    #Looks up a usergroup and returns the most common hashtag at a particular time & the probability of that hashtag
    
    return (ht, prob)