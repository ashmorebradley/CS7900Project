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
import math
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
learnedTopicsBigram = []

topicsKeyWords = []
topicsKeyWordsWeight = []
topicsKeyWordsBi = []
topicsKeyWordsWeightBi = []
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
    num_components=5
    
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
        tempList = []
        for kw in range(len(top_terms_list)):
            tempList.append(top_terms_key[kw][1])
            
        topicsKeyWordsWeight.append(tempList)
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
    
     #Repeat for bigrams
    tf_bigram  = CountVectorizer(max_df=0.95, ngram_range = (2, 2), min_df=2, max_features=None, stop_words=stopWords)
    tfBi = tf_bigram.fit_transform(content)
    bi_features_names = tf_bigram.get_feature_names()
    terms = bi_features_names
    num_components = 4
    ldaBi = LatentDirichletAllocation(n_components=num_components, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
    modelBi = ldaBi.fit(tfBi)

    bi_components = modelBi.components_

    for index, component in enumerate(bi_components):
        zipped = zip(terms, component)
        top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:7]
        #Keep this
        top_terms_list=list(dict(top_terms_key).keys())

        print("Topic "+str(index)+": ",top_terms_list)      #Debug statement. 

        topicsKeyWordsBi.append(top_terms_list)
        learnedTopicsBigram.append(str(top_terms_list[0]) + " or " + str(top_terms_list[1]))

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
    
###This needs to report what terms matched.
def countOccurances(content):
    keyWordCount = []
    for i in range(len(learnedTopics)): 
        col = []
        for j in range(len(topicsKeyWords[i])):
            col.append(0)
            
        keyWordCount.append(col) #Create a list of lists. 
    
    wordCount = {}
    #Count the occurances of a word.
    for kws in range(len(topicsKeyWords)):
        for word in range(len(topicsKeyWords[kws])):
            keyWordCount[kws][word] = content.count(topicsKeyWords[kws][word])

    #Got number of keyword in a tweet for each topic.
#    for w in content:
#        wordCount[w] = content.count(w)
        
    #Compare to known topics...
#    hits = 0
#    for i in range(len(topicsKeyWords)): #For every topic
#        for term in topicsKeyWords[i]:   #For every topic keyword.
#           # print(wordCount.keys())
#          #  print(term)
#            if term in wordCount.keys():
#                #print("HIT" + str(term))
#                hits += wordCount[term]
#        topicCount[i] = hits        

    return keyWordCount

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
    
    #Count number of words.
#    pd.value_counts(np.array(hold))
    #Count occurances of keywords
    wordCount = countOccurances(hold)
    #Got a list of hit of terms...
    #Convert to cosine  
    numWord = len(hold)
    
    for i in range(len(wordCount)):
        for j in range(len(wordCount[i])):
            wordCount[i][j] = wordCount[i][j] / numWord
        #Order large to small
    #    wordCount[i].sort(reverse = True)
    
    
    cosine = []    
    #All topics have been evaulated. Calculate the likelihood..

    for i in range(len(wordCount)):    #For every topic
        num = 0.0
        denomAs = 0.0
        denomBs = 0.0
        maxLen = min(len(wordCount), len(topicsKeyWordsWeight[i]) )     #This should always be the same but here to be safe.
        for j in range (maxLen):
            num += wordCount[i][j] * topicsKeyWordsWeight[i][j]
            denomAs += wordCount[i][j] * wordCount[i][j]
            denomBs += topicsKeyWordsWeight[i][j] * topicsKeyWordsWeight[i][j]
        
        if(num == 0):
            cosine.append(0)
        else:
            cosine.append(num / (math.sqrt(denomAs) * math.sqrt(denomBs)))    
            
        # probs[i] = topicCount[i] / len(hold)    #Probability is the number of overlaps / numwords
    #Add a chedk for NaN
    return cosine

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
    
    return (learnedTopics[loc])
      