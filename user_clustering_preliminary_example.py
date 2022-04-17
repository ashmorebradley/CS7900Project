import numpy as np
import pandas as pd

import datetime
# import time
from dateutil import tz

# year-mo-dy hh:mm:ss Time Zone Name

# extract data and convert to numpy arrays

df = pd.read_csv('MasterDataLocNoMessages.csv')

tweet_created_at = df['created_at']
tweet_username = df['username']
tweet_hashtags = df['hashtags']
tweet_geo = df['geo']


# convert to numpy arrays
tweet_created_at = tweet_created_at.to_list()
tweet_username = tweet_username.to_list()
tweet_hashtags = tweet_hashtags.to_list()
tweet_geo = tweet_geo.to_list()

# clean up arrays, converting strings to numbers

def extract_time(dt_str):
    '''
    Returns time in seconds since 1970 from an input date/time string
                    0123456789012345678
    string format: 'year-mo-dy hh:mm:ss Time Zone Name'
    '''
    year = int(dt_str[0:4])
    month = int(dt_str[5:7])
    day = int(dt_str[8:10])
    hour = int(dt_str[11:13])
    minute = int(dt_str[14:16])
    second = int(dt_str[17:19])
    EST = tz.gettz('America/New_York')
    datetime_obj = datetime.datetime(year, month, day, hour, minute, second, tzinfo = EST)
    seconds = datetime_obj.timestamp()
    return(seconds)


def extract_hashtags(ht_str):
    '''
    Returns a list of hashtags from an raw hashtag string
    '''
    if ht_str == '[]':
        return(np.array([]))
    else:
        ht_str = ht_str.translate({ord(c):None for c in "[]' "})
        ht_list = ht_str.split(',')
        ht_array = np.array(ht_list)
    return(ht_array)

def extract_geo(geo_str):
    '''
    Retures geo-location (lattitude and longitude) from a string extracted from tweet data
    '''
    geo_loc = geo_str.split(',')[:2]
    geo_loc = [float(loc) for loc in geo_loc]
    geo_loc = np.array(geo_loc)
    return(geo_loc)


# example_date = created_at[0]
# datetime_obj = extract_time(example_date)
# seconds = datetime_obj.timestamp()
# print('example_date: ', example_date)
# print('datetime_obj: ', datetime_obj)
# print('seconds: ', seconds)
#
#
# example_ht = hashtags[0]
# ht_array = extract_hashtags(example_ht)
# print('example_ht: ', example_ht)
# print('ht_array: ', ht_array)
#
#
# example_geo = geo[0]
# geo_loc = extract_geo(example_geo)
# print('example_geo: ', example_geo)
# print('geo_loc: ', geo_loc)



tweet_created_at = [extract_time(dt_str) for dt_str in tweet_created_at]
tweet_hashtags = [extract_hashtags(ht_str) for ht_str in tweet_hashtags]
tweet_geo = [extract_geo(geo_str) for geo_str in tweet_geo]


tweet_created_at = np.array(tweet_created_at)
tweet_username = np.array(tweet_username)
# tweet_hashtags = np.array(tweet_hashtags)
tweet_geo = np.array(tweet_geo)


# print('tweet_created_at: ', tweet_created_at)
# print('tweet_username: ', tweet_username)
# print('tweet_hashtags: ', tweet_hashtags)
# print('tweet_geo: ', tweet_geo)


all_hashtags = np.concatenate(tweet_hashtags)
# print('all_hashtags: ', all_hashtags)


unique_hashtags = np.unique(all_hashtags)
unique_usernames = np.unique(tweet_username)



print('number of hashtags: ', np.size(all_hashtags))
print('number of unique hashtags: ', np.size(unique_hashtags))
# print('unique_hashtags: ', unique_hashtags)


# print('tweet_hashtags: ', tweet_hashtags)

# find number of users who used each unique hashtag
unique_users = np.unique(tweet_username)
n_tweets = len(tweet_username)
n_users = len(unique_users)

print('n_tweets: ', n_tweets)
print('n_users: ', n_users)

# get all tweets of each user
hts_each_user = [np.array([])]*n_users
for tweet_idx, hts_array in enumerate(tweet_hashtags):
    username = tweet_username[tweet_idx]
    user_idx = np.flatnonzero(unique_users==username)[0]
    hts_each_user[user_idx] = np.concatenate([hts_each_user[user_idx], hts_array])

# reduce to unique tweets
for ii, user_hts in enumerate(hts_each_user):
    hts_each_user[ii] = np.unique(user_hts)


# keep only users with hashtags
n_hts_each_user = np.array([len(item) for item in hts_each_user])
keep_user_tf = n_hts_each_user > 0
unique_users = unique_users[keep_user_tf]
hts_each_user = np.array(hts_each_user)[keep_user_tf]
n_users = len(unique_users)
print('n_users: ', n_users)
print('hts_each_user: ', hts_each_user)

unique_hashtags
n_users_each_unique_hashtag = np.zeros(len(unique_hashtags))
for user_hashtags in hts_each_user:
    for hashtag in user_hashtags:
        hashtag_idx = np.flatnonzero(hashtag == unique_hashtags)[0]
        n_users_each_unique_hashtag[hashtag_idx] += 1

print('n_users_each_unique_hashtag: ', n_users_each_unique_hashtag)



# keep only hashtags used by multiple users
n_required_users = 2
keep_ht_tf = n_users_each_unique_hashtag > n_required_users
unique_hashtags = unique_hashtags[keep_ht_tf]
print('unique_hashtags: ', unique_hashtags)


# update user hashtags to only kept hashtags
for ii, user_hashtags in enumerate(hts_each_user):
    hts_each_user[ii] = np.intersect1d(unique_hashtags, user_hashtags)

# keep only users with shared hashtags
n_hts_each_user = np.array([len(item) for item in hts_each_user])
keep_user_tf = n_hts_each_user > 0
unique_users = unique_users[keep_user_tf]
hts_each_user = np.array(hts_each_user)[keep_user_tf]
n_users = len(unique_users)
print('n_users: ', n_users)
print('hts_each_user: ', hts_each_user)



# extract lattitudes and longitudes of these users (same indices)
print('unique_users: ', unique_users)

tweet_indices_for_user = { }
for username in unique_users:
    user_tweet_indices = np.flatnonzero(username == tweet_username)
    tweet_indices_for_user[username] = user_tweet_indices

print('tweet_username: ', tweet_username)
print('tweet_indices_for_user: ', tweet_indices_for_user)



# tweet_username =
# tweet_geo =

repeated_hts_each_user = { }
location_each_user = { }

for username in unique_users:
    user_tweet_indices = tweet_indices_for_user[username]
    location_each_user[username] = tweet_geo[user_tweet_indices[0]]
    repeated_hts = []
    for ii in user_tweet_indices:
        for ht in tweet_hashtags[ii]:
            if ht in unique_hashtags:
                repeated_hts.append(ht)
    repeated_hts_each_user[username] = np.array(repeated_hts)
print(5*'\n')

print('repeated_hts_each_user: ', repeated_hts_each_user)
print('location_each_user: ', location_each_user)
print('unique_hashtags: ', unique_hashtags)

# for each user, find percentage of tweets dealing with each unique shared hashtag

def get_user_data(username, repeated_hts_each_user, location_each_user):
    hashtags = repeated_hts_each_user[username]
    location = location_each_user[username]


unique_users
repeated_hts_each_user
location_each_user

for ii, username in enumerate(unique_users):
    user_hashtags = repeated_hts_each_user[username]
    ht_data = [np.sum(ht==user_hashtags) for ht in unique_hashtags]
    ht_data = np.array(ht_data)
    ht_data = ht_data/np.sum(ht_data)

    geo_data = location_each_user[username]

    print('ht_data: ', ht_data)
    print('geo_data: ', geo_data)

    user_data = np.concatenate([ht_data, geo_data])

    if ii == 0:
        EACH_USER_DATA = user_data
    else:
        EACH_USER_DATA = np.vstack([EACH_USER_DATA, user_data])

print('EACH_USER_DATA: ', EACH_USER_DATA)

# EACH_USER_DATA = [Pht1, Pht2, Pht3, lattitude, longitude]


from sklearn.preprocessing import MinMaxScaler
xscale_obj = MinMaxScaler(feature_range=(0, 1)).fit(EACH_USER_DATA)


EACH_USER_DATA_SCALED = xscale_obj.transform(EACH_USER_DATA)



# X_scaled = xscale_obj.transform(X_unscaled)
# Y_E2NN_pred = yscale_obj.inverse_transform(Y_E2NN_pred)
# Y_E2NN_pred = Y_E2NN_pred.flatten()


from sklearn.cluster import KMeans

k = 10
kmeans = KMeans(n_clusters = k, n_init=10)
cluster_idx = kmeans.fit_predict(EACH_USER_DATA_SCALED)
cluster_centers = kmeans.cluster_centers_

print('cluster_centers: ', cluster_centers)

# cluster_idx_new = kmeans.predict(NEW_USER_DATA)



from sklearn.metrics import silhouette_score
score = silhouette_score(EACH_USER_DATA_SCALED, kmeans.labels_)

print('score: ', score)


























#
