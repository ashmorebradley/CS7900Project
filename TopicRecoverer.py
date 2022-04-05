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
#date - A string or date time object
#topic - A string


def loadData(tweetId, user = 0, content, hashtag = 0, location = 0, date = 0, time = 0, topic = 0):
    #This funciton loads data into the topicDataRaw dictionary
    
    #content == document list.
    # Initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    
    stopWords = text.ENGLISH_STOP_WORDS.union({"http", "https", "Https", "Https:", "t", "s"})
    
#    print(stopWords)
 #   print("-----STOP WORDS----")
    # Vectorize document using TF-IDF
    tfidf = TfidfVectorizer(lowercase=True,
                            stop_words=stopWords,
                            ngram_range = (1,1),
                            tokenizer = tokenizer.tokenize)
    
    # Fit and Transform the documents
    train_data = tfidf.fit_transform(content)   
    
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
        print("Got topics")
        #Adding each top_term as a topic. This may be a bad idea.
        for t in top_terms_list:
            print("evaluating topic: " + str(t))
            for i in range(len(user)):
                if(t in content[i]):    #Topic is in tweet. Add to data.
                    if not t in topicDataRaw.keys():
                        topicDataRaw[t] = [(content[i], user[i], time[i], hashtag[i])]
                    else:
                        topicDataRaw[t].append( (content[i], user[i], time[i], hashtag[i]) )
    
    
    
    print("Done")
    
def processRawData():
    #This funciton process data in the topicDataRaw variable into a usable form and stores the result in topicDataProcessed
    
    
    print("Done")
    

def determineTopic(content):
    print("Done")    
    
#This fuction uses data in the topicDataProcessed variable to determine the most-likely topic.    
def determineTopic( User, hashtags, location, time):
    
    print("Done")