import pandas as pd
import reverse_geocoder as rg
from datetime import datetime

#read data from the csv file
data = pd.read_csv('tweetData.csv')

#storing the required data to a variable
geo_Locations = data['geo']
tweet_data = data['tweet']
hashtag_data=data['hashtags']
datetime_data=data['created_at']
date_data=data['date']


location_actual=[]

def reverseGeocode(coordinates):
    result = rg.search(coordinates)
    return result

def validateIncomingDate(date):
    incomingDate=date.split(' ')
    date_matched_data = []
    for idx, s in enumerate(date_data):
        if incomingDate[0] in s:
            date_matched_data.append(idx)
    return date_matched_data
 

    
def validateHashTagData(hashtag):
    hashTag_matched_data=[]
    for idx, d in enumerate(hashtag_data):
        if d is not None and len(d)>2:
            d_list=d.split(",")
            for s in d_list:
                for tag in hashtag:
                    if tag.upper() in s.upper():
                        hashTag_matched_data.append(idx)
        
    return hashTag_matched_data
    

def determineAuthor(Content, hashtag, location, time):
    content_matched_data=[]
    date_list=[]
    hashTag_data=[]
    
    if Content is not None and Content != "":
        for idx, s in enumerate(tweet_data):
            if Content in s:
                content_matched_data.append((idx))
    
    if time is not None and time != "":
        date_list = validateIncomingDate(time)
    
    if hashtag is not None and hashtag != "" and len(hashtag) > 0:
        hashTag_data=validateHashTagData(hashtag)
        
    print(content_matched_data)
    print(date_list)
    print(hashTag_data)
        
    common_in_all=set(content_matched_data) & set(date_list) & set(hashTag_data)
    
    print(common_in_all)
    
    if len(common_in_all) == 1:
        return data.iloc[[list(common_in_all).pop()]].get('username')
    elif len(content_matched_data) == 1:
        return data.iloc[[content_matched_data[0]]].get('username')
    elif len(date_list) == 1:
        return data.iloc[[date_list[0]]].get('username')
    elif len(hashTag_data) == 1:
        return data.iloc[[hashTag_data[0]]].get('username')
    elif location is not None:
        list_of_indexes=[]
        for s in content_matched_data:
            list_of_indexes.append(s)
        for s in date_list:
            list_of_indexes.append(s)
        for s in hashTag_data:
            list_of_indexes.append(s)
        res = []
        index_with_geo=[]
        for i in list_of_indexes:
            if i not in res:
                res.append(geo_Locations[i])
                index_with_geo.append((i,geo_Locations[i]))
                
        dist_res = []
        for i in res:
            if i not in dist_res:
                dist_res.append(i)
        result = []
        for t in dist_res:
            list_lat_long=t.split(",")
            coordinates =(list_lat_long[0],list_lat_long[1])
            address=reverseGeocode(coordinates)
            if address is not None and len(address)>0:
                result.append(address[0].get('name'))
        print(result)
        lat_long_of_input=""
        for idx, loc in enumerate(result):
            if loc is not None and location is not None and location.upper() == loc.upper():
                lat_long_of_input=dist_res[idx]
        location_data=[]
        for l in index_with_geo:
            if lat_long_of_input == l[1]:
                location_data.append(l[0])
        
        common_in_all_four=common_in_all & set(location_data)
        return data.iloc[[list(common_in_all_four).pop()]].get('username') 
