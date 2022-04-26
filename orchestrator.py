# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:37:43 2022

@author: ashmo
"""
#########################################
#Planer Agent
#########################################

#import PaperEmulator
import AuthorRecoverer
import TopicRecoverer
import UserAnalyzer
#import UserAnalyzer
import pandas as pd
#########################################
#Variables used by the agent
#########################################

logicTable = [
        #Content, Creator, Time, Hashtag, Location
        ([False, False, False, False, False], None , [False, False, False, False, False], "Nothing can be done"),       
        ([False, False, False, False, True], None, [True, False, False, False, True], "Risky topic recovery"    ),
        ([False, False, False, False, True], None, [False, False, False, True, True], "Risky hashtag recovery"    ),
        ([False, False, False, True, True], None, [False, True, False, True, True], "Risky Creator recovery from hashtag & Loc"    ),
        ([False, False, False, True, True], TopicRecoverer.determinTopic, [True, False, False, True, True], "Risky Topic recovery from hashtag & Loc "    ),
        ([False, True, False, True, False], TopicRecoverer.determinTopic, [True, True, False, True, False], "Reocver topic from creator and hashtag" ),
        ([False, True, False, False, False], TopicRecoverer.determinTopic, [True, True, False, False, False], "Recover topic from creator"    ),
        ([False, True, False, False, True], TopicRecoverer.determinTopic, [True, True, False, False, True], "Recover topic  from creator and location"    ),

        ([False, False, False, False, True], TopicRecoverer.determinTopic, [True, False, False, False, True], "Recover topic from location"    ),        
        ([False, False, False, True, False], TopicRecoverer.determinTopic, [True, False, False, True, False], "Recover topic from hashtag"    ),
        ([False, False, False, True, True] , TopicRecoverer.determinTopic, [True, False, False, True, True], "Reocver topic from hashtag and location"   ),
#        ([False, False, True, False, False]    ),
        ([False, False, True, False, True], None, [False, False, True, True, True], "Recover Hashtag from time and location"   ),
        ([False, False, True, False, True], None, [False, True, True, False, True], "Recover User from time and location"   ),
        ([False, False, True, False, True], TopicRecoverer.determinTopic, [True, False, True, False, True], "Recover topic from time and location"   ),
 
        ([False, True, False, False, True], None, [False, True, False, True, True], "Recover hashtag from creator & location"   ),
#        ([False, True, False, True, False]    ),
        ([False, True, False, True, True], TopicRecoverer.determinTopic, [True, True, False, True, True], "Recover content from creator, hash, & location"   ),
#        ([False, True, True, False, False]    ),
#        ([False, True, True, False, True]    ),
#        ([False, True, True, True, False]    ),
#        ([False, True, True, True, True]    ),
        ([True, False, False, False, False], AuthorRecoverer.determineAuthor, [True, True, False, False, True], "Recover Author - Uncertain"),
        ([True, False, False, False, True], AuthorRecoverer.determineAuthor, [True, True, False, False, True], "Recover Author"),         
        ([True, False, False, True, False], AuthorRecoverer.determineAuthor, [True, True, False, True, False], "Recover Author"   ),         
        ([True, False, False, True, True] , AuthorRecoverer.determineAuthor, [True, True, False, True, True], "Recover Author"  ),         
        ([True, False, True, False, False], AuthorRecoverer.determineAuthor, [True, True, True, False, False], "Recover Author"   ),         
        ([True, False, True, False, False], AuthorRecoverer.determineAuthor, [True, False, True, True, False], "Recover Hash"   ),        
        ([True, False, True, False, True], AuthorRecoverer.determineAuthor,  [True, False, True, True, True], "Recover Hash"   ),          
        ([True, False, True, False, True] , AuthorRecoverer.determineAuthor,[True, True, True, False, True], "Recover Author"  ),         
        ([True, False, True, True, False] , AuthorRecoverer.determineAuthor, [True, True, True, True, False], "Recover Author"  ),         
        ([True, False, True, True, True]  , AuthorRecoverer.determineAuthor, [True, True, True, True, True], "Recover Author" ),         
        ([True, True, False, False, False], UserAnalyzer.commonHashtags, [True, True, False, True, False], "Generating Hashtag"),         
        ([True, True, False, False, True], UserAnalyzer.commonHashtags, [True, True, False, True, True], "Generating Hashtag"),         
        ([True, True, False, True, False], UserAnalyzer.recoverTime, [True, True, True, True, False], "Get Time" ),         
        ([True, True, False, True, True], UserAnalyzer.recoverTime, [True, True, True, True, True], "Estimate Time"),         
        ([True, True, True, False, False], UserAnalyzer.commonHashtags, [True, True, True, True, False],  "Recover Hashtag" ),         
        ([True, True, True, False, True], UserAnalyzer.commonHashtags, [True, True, True, True, True], "Recover Hashtag" ),         
        ([True, True, True, True, False], UserAnalyzer.commonLocation, [True, True, True, True, True], "Recover Location"),
        ([True, True, True, False, False], UserAnalyzer.commonLocation, [True, True, True, False, True], "Recover Location"),
        ([True, True, False, True, False], UserAnalyzer.commonLocation, [True, True, False, True, True], "Recover Location"),
        ([False, True, True, True, False], UserAnalyzer.commonLocation, [False, True, True, True, True], "Recover Location"),
        ([False, True, True, False, False], UserAnalyzer.commonLocation, [False, True, True, False, True], "Recover Location"),
        ([False, True, False, True, False], UserAnalyzer.commonLocation, [False, True, False, True, True], "Recover Location"),
        ([False, True, True, False, False], UserAnalyzer.commonLocation, [False, True, True, False, True], "Recover Location"),
        ([False, True, True, True, False], UserAnalyzer.commonLocation, [False, True, True, True, True], "Recover Location"),
        ([False, True, False, False, False], UserAnalyzer.commonLocation, [False, True, False, False, True], "Recover Location"),
        
        ]
#Int the form [Condition, action, endstate] 
tableData = {'Content': [.011, .02, .025, .033, .04 ], 'User': [.001, .02, .021, .022, .033 ], 'Time': [.45, .57, .64, .72, .074 ], 'Hash': [.011, .015, .021, .029, .04 ], 'Loc':[.001, .01, .015, .021, .04 ]}
uncertTable = pd.DataFrame(tableData)

probabilityTable = {}

#print(logicTable)
def compareDics(dicA, dicB):
    k = dicA.keys()
    if(k != dicB.keys()):
        return False
    
    for el in k:
        if(dicA[el] != dicB[el]):
            return False
        
    return True



#UserHash is a username-indexed dictionary. Each user name points to a list of tuples containing topic, time, and a list of hashtags)
#{User ==> ([ [topic, time, [hashtag]] ] )}
#I invision each tweet to be its own entry in the list.

internalTweet = {}
iState = [False, False, True, False, True]

stateDecode = ['Content', 'User', 'Time', 'Hash', 'Loc']

def stateToString(s):
    out = ""
    count = 0
    for i in s:
        if(i):
            if(count == 0):
                out = out + "Content, "
            elif(count == 1):
                out = out + "User, "  
            elif(count == 2):
                out = out + "Time, "
            elif(count == 3):
                out = out + "Hash, "
            elif (count == 4):
                out = out + "Loc "    

        else:
            if(count == 0):
                out = out + "!C___, "
            elif(count == 1):
                out = out + "!U___, "  
            elif(count == 2):
                out = out + "!T___, "
            elif(count == 3):
                out = out + "!H___, "
            elif (count == 4):
                out = out + "!L___"    
            
        count = count + 1
    return out

    
class Node:
    def __init__(self, state, goal, action = None, txt = ""):
        self.children = []
        self.action = action
        self.state = state
        self.goal = goal
        self.actionText = txt
        
    def addChild(self, c):
        self.children.append(c)
        
    def goToChild(self, x):
        self.children[x].action()
        return self.childern[x]
   
    def isLeaf(self):
        if len(self.children) == 0:
            return True
        return False

        
    def print(self, offset, depth = 0):
        temp = depth
        lead = ""
        while(temp > 0):
            lead = lead + "|" 
            if(temp == 1):
                lead = lead + '_'
            else:
                lead = lead + offset
            temp -= 1
            
        print(lead +  " Action :" + str(self.action)  + " - State :" + stateToString(self.goal))
        if(depth == 0):
            print("|")
        else:
            print(lead[0:len(lead) -1])
        for c in self.children:
            c.print((len(offset) + 1) * " ", depth + 1)

#Need to add actoins to the nodes.
def makePlanTree(state, goal):
#Node constains from state to goal state.
    plan = Node(state, goal)
    nodeQueue = []
    nodeQueue.append(plan)
    #Find goal in table above
    while nodeQueue:    #While not empty
        n = nodeQueue.pop(0)
       
        for row in logicTable:
         #   print(row)
           # print("Looking for " + stateToString(n.goal) + " vs. " + stateToString(row[2]))    
            if(row[2] == n.goal):
          #      print("\tis equal")
                #Find position difference
                diffs = []
                for i in range(len(row[2])):
                    if(row[0][i] != row[2][i]):
           #             print("!!!Diff found")
                        diffs.append(i)
                for i in diffs:
                    if state[i] != goal[i]:
            #            print("Adding State : " + str(state) + " Goal : " + str(row[0]) + " Action: " + str(row[1]))
                        n.addChild(Node( state, row[0], action = row[1], txt = row[3]))
              #      else:
             #           print("Diff and goal are the same!!!!")
                
        for c in n.children:
            nodeQueue.append(c)
        
    #    plan.print("")
     #   print("")
   # plan.print("")    
    return plan    
    #Condition state becomes the new goal
    #Repeat until current state is reached or fail.

#Performs a depth-first search to find all possible plans then checks for feasibility.
def findFeasible(root, state, goal):
    finalPlans = []  #This becomes the list of all possible plans.
    currentPlan = []
    if root is None:    #Check for complete.
        return finalPlans
    
    nodeStack = []
    nodeStack.append(root)
    
    while nodeStack:    #While there are nodes in the stack.
        r = nodeStack.pop()
        currentPlan.append( (r.action, r.goal, r.actionText) )    #Add action to the current plan
        if(r.isLeaf()): #Check for an end
            finalPlans.append(currentPlan.copy())
            currentPlan.pop()   #Remove the last entry 
        else: #Add non-leaf nodes to the stack.
            for n in r.children:
                 nodeStack.append(n)
   
    #Now have a list of candidate plans. Find the ones that work.
    keepList = []
    for plan in range(len(finalPlans)):
        if isValid(finalPlans[plan], state, goal):
            keepList.append(finalPlans[plan])
        
    return keepList

#Plan is a list of touples containing actions, endstates. 
#Follow the endstates to make sure on one variable changes at a time.
def isValid(plan, state, goal):
    
    #Check start state and end state
    l = len(plan)
    if plan[0][1] != goal:
        return False
    
    if plan[l-1][1] != state:  #Wrong starting state
        return False
    

    for i in range(len(plan) - 1):
        cur = plan[i][1]
        nxt = plan[i+1][1]
        difs = 0
        for j in range(len(cur)):
            if cur[j] != nxt[j]:
                difs += 1
        
        #End of for-loop
        if difs != 1:
            return False
            
    return True
    
#Returs a list of tuples (function, newgoalState, string descripiton)    
def makePlan(state, goal):
    candidates = makePlanTree(state, goal)
    out = findFeasible(candidates, state, goal)
 #   print(out)
    print("---------")
    
    return out

def printSinglePlan(plan):
    plan.reverse()
    for step in plan[:-1]:
        print("\t" + str(step[2]) + " Using " + str(step[0]))

def printPlans(planArr):
    counter = 0
    for plan in planArr:
        print("###Printing plan " + str(counter + 1) + " of " + str(len(planArr)))
        counter += 1
        plan.reverse()
        for step in plan[:-1]:
            print("\t" + str(step[2]) + " Using " + str(step[0]))
        print("")
            
def selectPlan(plans):
 #   print("NumPlans " + str(len(plans)))
    for i in range(len(plans)):
        plans[i].reverse()
    
    best = 9999
    diff = ""
    uncert = [0.0] * len(plans)  
    print("")
    
    for i in range(len(plans)): #For every plan
      #  print("I is : " + str(i))
        numSteps = len(plans[i])
        for jj in range(numSteps - 1):  #For every step (Tuple)
            #Find the diff.
            for k in range(len(plans[i][jj][1])):
                if(plans[i][jj][1][k] != plans[i][jj+1][1][k]):
                    diff = stateDecode[k]
            #Diff found, calculate uncert.
            vUsed = 4
            if numSteps < 4:
                vUsed = numSteps
                
            uncert[i] += uncertTable.iloc[vUsed - jj].loc[diff]
            
        if best > uncert[i]:
            best = uncert[i]
        
  #      print("Uncertainty measure in plan " + str(i + 1) + ": " + str(uncert[i]))        
    #Find best 
    for i in range(len(uncert)):
        if best == uncert[i]:
   #         print("")
    #        print("-----------Final Plan----------")
     #       print("Secected plan " + str(i+1))
            return plans[i]
        
    print("FAIL!!!!")
    return plans[0]
    


#print(len(result[0]))
        
#This function executes an existing plan. This will also record probability.
#Startparams is a dictionary of tweet fields.
def executePlan(plan, startParams):
    reconstructedTweet = startParams
    
    for step in plan:
        #Determine parameters.
        #Call funciton.    
        
        result = step[2](reconstructedTweet)
        
    #Collect result.
        for j in range(len(step[1])):
            if step[1][i] == step[3][i]:
                if i == 1:
                    reconstructedTweet['content'] = result
                elif i == 2:    
                    reconstructedTweet['creator'] = result
                elif i == 3:
                    reconstructedTweet['time'] = result
                elif i == 4:
                    reconstructedTweet['hashtag'] = result
                elif i == 5:
                    reconstructedTweet['location'] = result
    #Find table entry.
    #Update belief value.
    return reconstructedTweet
    
    
def setActiveTweet(t):
     vals = keys(t)
     #This will need to be standardized.
     keyList = ["Content", "Creator", "Time", "Hashtag", "Location"]
     for el in keyList:
         internalTweet[el] = keys[el]
         if(el in vals):
             state[el] = True
         else:
             state[el] = False
             
    
