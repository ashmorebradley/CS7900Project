# import numpy as np
# import pandas as pd
#
# import datetime
# from dateutil import tz
#
# import pickle










def GenerateUserGroups(filename='MasterDataLocNoMessages.csv'):

    import numpy as np
    import pandas as pd

    import datetime
    from dateutil import tz

    import pickle

    # extract data and convert to numpy arrays
    df = pd.read_csv(filename)

    tweet_created_at = df['created_at']
    tweet_username = df['username']
    tweet_hashtags = df['hashtags']
    tweet_geo = df['geo']


    # convert from pandas to lists
    tweet_created_at = tweet_created_at.to_list()
    tweet_username = tweet_username.to_list()
    tweet_hashtags = tweet_hashtags.to_list()
    tweet_geo = tweet_geo.to_list()

    # functions to clean up arrays, converting strings to numbers
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

    # extract information from strings
    tweet_created_at = [extract_time(dt_str) for dt_str in tweet_created_at]
    tweet_hashtags = [extract_hashtags(ht_str) for ht_str in tweet_hashtags]
    tweet_geo = [extract_geo(geo_str) for geo_str in tweet_geo]

    # convert to numpy arrays
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

    # print('tweet_username: ', tweet_username)
    # print('tweet_indices_for_user: ', tweet_indices_for_user)


    # extract hashtags (including multiple uses) and locations of kept users
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


    print('repeated_hts_each_user: ', repeated_hts_each_user)
    print('location_each_user: ', location_each_user)
    print('unique_hashtags: ', unique_hashtags)


    # for each user, find percentage of tweets dealing with each unique shared
    # hashtag, as well as lattitude and longitude information

    for ii, username in enumerate(unique_users):
        user_hashtags = repeated_hts_each_user[username]
        # number of times each unique hashtag appears
        ht_data = [np.sum(ht==user_hashtags) for ht in unique_hashtags]
        ht_data = np.array(ht_data)
        # fraction of the time each unique hashtag appears
        ht_data = ht_data/np.sum(ht_data)

        # location data
        geo_data = location_each_user[username]

        # print('ht_data: ', ht_data)
        # print('geo_data: ', geo_data)

        user_data = np.concatenate([ht_data, geo_data])

        if ii == 0:
            EACH_USER_DATA = user_data
        else:
            EACH_USER_DATA = np.vstack([EACH_USER_DATA, user_data])

    print('EACH_USER_DATA: ', EACH_USER_DATA)

    # EACH_USER_DATA = [Pht1, Pht2, Pht3, lattitude, longitude]

    # scale data
    from sklearn.preprocessing import MinMaxScaler

    xscale_obj = MinMaxScaler(feature_range=(0, 1)).fit(EACH_USER_DATA)
    EACH_USER_DATA_SCALED = xscale_obj.transform(EACH_USER_DATA)



    # X_scaled = xscale_obj.transform(X_unscaled)
    # Y_E2NN_pred = yscale_obj.inverse_transform(Y_E2NN_pred)
    # Y_E2NN_pred = Y_E2NN_pred.flatten()


    from sklearn.cluster import KMeans

    # use between 2 and 8 clusters
    k_options = np.array(range(2,9))

    scores = np.zeros(np.shape(k_options))

    for idx, k in enumerate(k_options):

        k = 10
        kmeans = KMeans(n_clusters = k, n_init=10)
        cluster_idx = kmeans.fit_predict(EACH_USER_DATA_SCALED)
        cluster_centers = kmeans.cluster_centers_

        # print('cluster_centers: ', cluster_centers)

        from sklearn.metrics import silhouette_score
        score = silhouette_score(EACH_USER_DATA_SCALED, kmeans.labels_)

        # print('score: ', score)

        scores[idx] = score
    #

    opt_idx = np.argmax(scores)

    k_opt = k_options[opt_idx]
    opt_score = scores[opt_idx]
    print('k_opt: ', k_opt)
    print('opt_score: ', opt_score)

    from matplotlib import pyplot as plt

    fig1 = plt.figure(figsize=(6.4, 4.8))
    ax = fig1.add_subplot(111)
    # plot cities
    ax.plot(k_options, scores, 'ko-', linewidth=2)#, label='')
    ax.set_title('silhouette scores')
    ax.set_xlabel('k clusters', fontsize=18)
    ax.set_ylabel('score', fontsize=18)
    # ax.legend(loc='upper right')
    # save_fig('silhouette scores')
    plt.draw()
    fig1.show()
    input(['Press enter to continue'])
    plt.close(fig1)



    # save clustering info
    usergroups_filename = 'clusters_info.pkl'
    usergroup_info = {'kmeans': kmeans,
                      'xscale_obj': xscale_obj,
                      'unique_hashtags': unique_hashtags,
                      'unique_users': unique_users,
                      'tweet_indices_for_user': tweet_indices_for_user,
                      'repeated_hts_each_user': repeated_hts_each_user,
                      'location_each_user': location_each_user}


    outfile = open(usergroups_filename, 'wb')
    pickle.dump(usergroup_info, outfile)
    outfile.close()




def ClusterGivenUsers(user_hashtags, geo_array):
    '''
    user_hashtags_array: a list or numpy array of strings which are hashtags used by the user
    geo_array: a numpy array of the lattitude and longitude from which the user tweets
    '''

    import numpy as np
    # import pandas as pd

    # import datetime
    # from dateutil import tz

    import pickle


    # loading and unpacking user and clustering information
    usergroups_filename = 'clusters_info.pkl'
    infile = open(usergroups_filename, 'rb')
    usergroup_info = pickle.load(infile)
    infile.close()

    kmeans = usergroup_info['kmeans']
    xscale_obj = usergroup_info['xscale_obj']
    unique_hashtags = usergroup_info['unique_hashtags']
    unique_users = ['unique_users']
    tweet_indices_for_user = ['tweet_indices_for_user']
    repeated_hts_each_user = ['repeated_hts_each_user']
    location_each_user = ['location_each_user']

    # get index of cluster user belongs to
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans

    user_hashtags = np.array(user_hashtags)
    # get number of times each hashtag appears
    ht_data = [np.sum(ht==user_hashtags) for ht in unique_hashtags]
    # get percentage of times each hashtag is used
    ht_data = ht_data/np.sum(ht_data)
    NEW_USER_DATA = np.concatenate([ht_data, geo_array])

    NEW_USER_DATA_SCALED = xscale_obj.transform(NEW_USER_DATA.reshape(1, -1))

    cluster_idx_new = kmeans.predict(NEW_USER_DATA_SCALED)

    cluster_centers = kmeans.cluster_centers_
    cluster_center = cluster_centers[cluster_idx_new, :]
    print('cluster_idx_new: ', cluster_idx_new)
    print('cluster_center: ', cluster_center)

    cluster_center_descale = xscale_obj.inverse_transform(cluster_center.reshape(1, -1))
    print('cluster_center_descale: ', cluster_center_descale)
    print('unique_hashtags: ', unique_hashtags)

    return(cluster_idx_new, cluster_center_descale)

if __name__ == "__main__":
    import numpy as np
    GenerateUserGroups()
    user_hashtags = np.array(['amc','ukraine','iealfjoiea'])
    user_lattitude = 37
    user_longitude = -90
    geo_array = np.array([user_lattitude, user_longitude])
    ClusterGivenUsers(user_hashtags, geo_array)



















#
