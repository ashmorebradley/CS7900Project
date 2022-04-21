# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 00:58:28 2022

@author: ashmo
"""

#########################################
#PartialTopicAssigner Agent
#########################################
import os
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
#########################################
#Variables used by the agent
#########################################
topicDataRaw = {}
#topicData  is a topic-indexed dictionary. Each topic points to a list of tuples containing content, user, time, and a list of hashtags)

topicDataProcessed = {}
#topicData is a topic-indexed dictionary. Each topic name points to a list of statistics that can be used to decide the most-likely topic of a deleted tweet.
#Contents of this diciton ary are TBD.

learnedTopics = []

topicsKeyWords = []

#########################################
#Function used to add data to the agent.
#User - A string.
#Content - A string
#Hashtag - A string
#locaiton - A tuple of lat/lon values, floats
#time - A string or date time object
#date - A string or date time object
#topic - A string

########################################################################
#Accepts a collection of tweets and determines the keywords for topics.
#This function has one hyperparameter, num_components. This is the number of topics that will be determined.
########################################################################
def loadData(tweetId, user = 0, content = 0, hashtag = 0, location = 0, date = 0, time = 0, topic = 0):
    #This funciton loads data into the topicDataRaw dictionary
    
    #content == document list.
    # Initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    
    stopWords = text.ENGLISH_STOP_WORDS.union({"http", "https", "Https", "Https:", "t", "s"})
    
    # Vectorize document using TF-IDF
    tfidf = TfidfVectorizer(lowercase=True,
                            stop_words=stopWords,
                            ngram_range = (1,1),
                            tokenizer = tokenizer.tokenize)
    
    # Fit and Transform the documents
    train_data = tfidf.fit_transform(content)   
    
    ######Hyperparameter####
    # Define the number of topics or components
    num_components=4
    
    # Create LDA object
    model=LatentDirichletAllocation(n_components=num_components)
    
    # Fit and Transform SVD model on data
    lda_matrix = model.fit_transform(train_data)
    
    # Get Components 
    lda_components=model.components_
    
    terms = tfidf.get_feature_names()
    
    #Now have a list of topics. Propigate in dictionary. 
    for index, component in enumerate(lda_components):
        zipped = zip(terms, component)
        top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:7]
        top_terms_list=list(dict(top_terms_key).keys())
#        print("Topic "+str(index)+": ",top_terms_list)      #Debug statement. 
     
        topicsKeyWords.append(top_terms_list)
        learnedTopics.append(str(top_terms_list[0]) + " or " + str(top_terms_list[1]))
        #Adding each top_term as a topic. This may be a bad idea.
#        for t in top_terms_list:
#            print("evaluating topic: " + str(t))
#            for i in range(len(user)):
#                if(t in content[i]):    #Topic is in tweet. Add to data.
#                    if not t in topicDataRaw.keys():
#                        topicDataRaw[t] = [(content[i], user[i], time[i], hashtag[i])]
#                    else:
#                        topicDataRaw[t].append( (content[i], user[i], time[i], hashtag[i]) )
#    
#    
    
    print("Done")
    #print(learnedTopics)
    
########################################################################    
#
########################################################################       
def removeStopwords(l, stops):
    toReturn = l
    for word in l:
        for s in stops:
            if(word == s):
                toReturn.remove(word)
                
    return toReturn
    
########################################################################    
#Returns number of words in a tweet that align with a known topic.
#Used to assign topics, not ment to be called externally.
#Content - A list of terms.
########################################################################
def countHits(content):
    topicCount = [0] * len(learnedTopics)
    
    wordCount = {}
    #Count the occurances of a word.
    for w in content:
        wordCount[w] = content.count(w)
        
    #Compare to known topics...
    hits = 0
    for i in range(len(topicsKeyWords)): #For every topic
        for term in topicsKeyWords[i]:   #For every topic keyword.
           # print(wordCount.keys())
          #  print(term)
            if term in wordCount.keys():
                #print("HIT" + str(term))
                hits += wordCount[term]
        topicCount[i] = hits        

    return topicCount

########################################################################
#
########################################################################    
def determineTopicProbs(content):
    #Initialize variable
    probs = [0.0] * len(learnedTopics)
    
    #Tokenize
    hold = content.split()
    for i in range(len(hold)):
        hold[i] = hold[i].lower()
        
    stopWords = text.ENGLISH_STOP_WORDS.union({"http", "https", "Https", "Https:", "t", "s"})
    hold = removeStopwords(hold, stopWords)
    
    
    topicCount = countHits(hold)
    
    #All topics have been evaulated. Calculate the likelihood..
    for i in range(len(topicCount)):
        probs[i] = topicCount[i] / len(hold)    #Probability is the number of overlaps / numwords
    
    return probs

########################################################################
#Returns the topics identified by the agent.
########################################################################    
def getKnownTopics():
   return learnedTopics

########################################################################
#Determines the topic of a tweet's content.
#This will compare to know topics thus load_data() must be called first!
########################################################################
def determineTopic(content):
    probs = determineTopicProbs(content)

    #Find max
    mx = 0
    loc = 0
    for i in range(len(probs)):
        if(probs[i] > mx):
            mx = probs[i]
            loc = i
    
    return (learnedTopics[i])
      