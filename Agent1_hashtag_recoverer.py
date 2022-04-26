

def RecoverHashtag(lattitude, longitude, post_time, filename='MasterDataLocNoMessages.csv'):
    '''
    Given the location and time of a tweet, this agent returns likely canidates for
    hashtags contained in the tweet, and the probability of each hashtag.

    The probabilities assume the tweet only contains a single hashtag
    (Many tweets contain no hashtags, or multiple hashtags.)

    This agent uses a database of tweet hashtags, locations, and times.
    '''

    import numpy as np
    from numpy.random import default_rng
    rng = default_rng()

    import pandas as pd

    import datetime
    from dateutil import tz

    import pickle

    # convert time to seconds if it is a string
    if isinstance(post_time, str):
        dt_str = post_time
        year = int(dt_str[0:4])
        month = int(dt_str[5:7])
        day = int(dt_str[8:10])
        hour = int(dt_str[11:13])
        minute = int(dt_str[14:16])
        second = int(dt_str[17:19])
        EST = tz.gettz('America/New_York')
        datetime_obj = datetime.datetime(year, month, day, hour, minute, second, tzinfo = EST)
        seconds = datetime_obj.timestamp()
        post_time = seconds


    '''
    Steps:
    0) determine needed information
        a) created_at, geo, hashtags
    1) determine format used in this file
       a) created_at:
       b) geo:
       c) hashtags:
       d) X: numpy array of matrics [ht_idx, lattitude, longitude, time]
    2) determine format used in user clustering
        a) created_at: numpy array of seconds since 1970
        b) geo: numpy array of numpy arrays of coordinates
        c) hashtags: list of numpy arrays of tweet hashtags
    3) Convert from format 2) to format 1)
        i) need list of hashtags w/ > 10 uses that can be called using indices
    4) Add noise to locations (uniform^2*50 km, uniform direction)
    5) scale data and save scaling
    6) Divide up data by hashtag
    7) plot data
    8) Fit GMMs
    9) complete analysis
    10) complete plotting
    11) output predicted possibilities for result

    '''

    # try to load clustering information for hashtags
    try:

        from sklearn.preprocessing import MinMaxScaler

        # load scale object
        scale_filename = "data_scale_object_agent_1.pkl"
        infile = open(scale_filename, 'rb')
        xscale_obj = pickle.load(infile)
        infile.close()

        from sklearn.mixture import BayesianGaussianMixture
        from scipy.stats import multivariate_normal

        # load clustering info
        model_filename = "BGMM_models_agent_1.pkl"
        infile = open(model_filename, 'rb')
        hashtag_mixture_models = pickle.load(infile)
        infile.close()

        # load hashtag info
        hashtag_filename = "hashtag_info_agent_1.pkl"
        infile = open(hashtag_filename, 'rb')
        hashtag_info = pickle.load(infile)
        infile.close()

        unique_hashtags = hashtag_info["unique_hashtags"]
        N_HT_INSTANCES = hashtag_info["N_HT_INSTANCES"]
        N_HT = len(unique_hashtags)


        ################################################################################
        ''' Get prior probabilites of each hashtag '''
        # based on urn problem (how frequent observations have been):
        P_prior_hashtags = (N_HT_INSTANCES + 1)/(np.sum(N_HT_INSTANCES)+N_HT)
        print('probabilities sum: ', np.sum(P_prior_hashtags))


        def pdf_gaussian_mixture(bgm, x):
            # Calculate the probability density function of a gaussian mixture model
            # for point or points x
            weights = bgm.weights_
            means = bgm.means_
            cov = bgm.covariances_

            n_components = len(weights)

            pdf = 0.
            for jj in range(n_components):
                rv = multivariate_normal(means[jj], cov[jj])
                gauss_pdf = rv.pdf(x)
                pdf += weights[jj] * gauss_pdf
            return(pdf)


        def hashtag_probabilities_bayes(x_predict):

            if len(x_predict.shape) == 1: # 1D array
                N_pred = 1
            else:
                N_pred = x_predict.shape[0]

            P_data_given_hts = np.zeros([N_pred, N_HT]) # Npredictions by Nhashtags
            for ii in range(len(hashtag_mixture_models)):
                bgm = hashtag_mixture_models[ii]
                pdf = pdf_gaussian_mixture(bgm, x_predict)
                P_data_given_hts[:,ii] = pdf

            P_ratio = P_prior_hashtags * P_data_given_hts # Npredictions by Nhashtags

            divisor = np.sum(P_ratio, 1)
            divisor = divisor.reshape([-1, 1])
            P_posterior = P_ratio / divisor # Npredictions by Nhashtags
            return(P_posterior)


        ################################################################################
        ''' Predict at requested point '''
        Xpredict = np.array([[lattitude, longitude, post_time]])
        Xpredict = xscale_obj.transform(Xpredict)


        probabilities = hashtag_probabilities_bayes(Xpredict)
        probabilities = probabilities[0]

        print("probabilities: ", probabilities)
        print("unique_hashtags: ", unique_hashtags)

        sort_idx = np.argsort(probabilities)[::-1]

        sorted_hashtags = unique_hashtags[sort_idx]
        sorted_probabilities = probabilities[sort_idx]
        print("sorted_probabilities: ", sorted_probabilities)
        print("unique_hashtags: ", unique_hashtags)
        return(sorted_hashtags, sorted_probabilities)



    # clustering information for hashtags not available
    # running clustering analysis and making relevant plots
    except:







        # extract data and convert to numpy arrays

        df = pd.read_csv(filename)

        tweet_created_at = df['created_at']
        tweet_hashtags = df['hashtags']
        tweet_geo = df['geo']

        # convert from pandas to lists
        tweet_created_at = tweet_created_at.to_list()
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
        # tweet_hashtags = np.array(tweet_hashtags)
        tweet_geo = np.array(tweet_geo)


        print('tweet_created_at: ', tweet_created_at)
        print('tweet_hashtags: ', tweet_hashtags)
        print('tweet_geo: ', tweet_geo)



        # consolidate information into 2D data matrix
        Xdata = np.array([])
        Xhashtags = np.array([])

        for ii, ht_array in enumerate(tweet_hashtags):
            for ht in ht_array:
                Xrow = np.hstack([tweet_geo[ii], tweet_created_at[ii]])
                Xrow = Xrow.reshape(1,-1) # make 2D
                if Xdata.size:
                    Xdata = np.vstack([Xdata, Xrow])
                    Xhashtags = np.vstack([Xhashtags, ht])
                else:
                    Xdata = Xrow
                    Xhashtags = np.array([[ht]])
        print('Xdata: ', Xdata)
        print('Xhashtags: ', Xhashtags)

        # get unique hashtags
        unique_hashtags = np.unique(Xhashtags)
        print('unique_hashtags: ', unique_hashtags)

        # eliminate uncommon unique hashtags
        N_ht_instances = np.zeros(np.shape(unique_hashtags))
        for ii, unique_hashtag in enumerate(unique_hashtags):
            count = np.sum(unique_hashtag==Xhashtags)
            N_ht_instances[ii] = count
        print('N_ht_instances: ', N_ht_instances)

        significant_hashtags = 10
        keep_idx = np.flatnonzero(N_ht_instances >= significant_hashtags)

        unique_hashtags = unique_hashtags[keep_idx]

        # Eliminate uncommon hashtag rows in image

        X_data_cleaned = np.array([])
        Xhashtags_keep = []
        for ii in range(np.shape(Xdata)[0]):
            if Xhashtags[ii] in unique_hashtags:
                Xrow = Xdata[ii,:]
                Xrow = Xrow.reshape(1,-1) # make 2D
                if X_data_cleaned.size:
                    X_data_cleaned = np.vstack([X_data_cleaned, Xrow])
                else:
                    X_data_cleaned = Xrow
                Xhashtags_keep.append(Xhashtags[ii])

        Xhashtags = np.array(Xhashtags_keep)
        print('unique_hashtags: ', unique_hashtags)
        print('X_data_cleaned: ', X_data_cleaned)
        print('Xhashtags: ', Xhashtags)



        # for ii, hashtag in enumerate(Xdata):
        #     if hashtag in unique_hashtags:



        # add small noise to locations (stability for GMM)
        lat_long_range = 0.7 # around 50 miles
        Ndata = np.shape(X_data_cleaned)[0]
        distances_rand = lat_long_range*rng.random([Ndata, 1])
        directions_rand = 2*np.pi*rng.random([Ndata, 1])

        sind = lambda x : np.sin(np.radians(x))
        cosd = lambda x : np.cos(np.radians(x))

        latt_long_noise = np.hstack([distances_rand*cosd(directions_rand), distances_rand*sind(directions_rand)])
        print('latt_long_noise: ', latt_long_noise)

        latt_long = X_data_cleaned[:, :2]

        print('latt_long: ', latt_long)
        print('latt_long_noise.shape: ', latt_long_noise.shape)
        print('latt_long.shape: ', latt_long.shape)

        latt_long = latt_long + latt_long_noise
        X_data_cleaned[:, :2] = latt_long
        print('X_data_cleaned: ', X_data_cleaned)


        # scale data (stability for GMM)
        from sklearn.preprocessing import MinMaxScaler

        xscale_obj = MinMaxScaler(feature_range=(0, 1)).fit(X_data_cleaned)
        X_data_scaled = xscale_obj.transform(X_data_cleaned)

        # save and load scale object
        scale_filename = "data_scale_object_agent_1.pkl"
        # save
        outfile = open(scale_filename, 'wb')
        pickle.dump(xscale_obj, outfile)
        outfile.close()
        # load
        infile = open(scale_filename, 'rb')
        xscale_obj = pickle.load(infile)
        infile.close()


        print('X_data_scaled: ', X_data_scaled)


        # separate out data by hashtag
        # X_hashtags = X_data_scaled[:,0]

        tag_idxs = []
        X_DATA_BY_HASHTAG = []
        N_HT = len(unique_hashtags)
        for ii, hashtag in enumerate(unique_hashtags):
            tag_idxs = np.flatnonzero(hashtag==Xhashtags)
            print("Xhashtags.shape: ", Xhashtags.shape)
            print("X_data_scaled.shape: ", X_data_scaled.shape)
            print("tag_idxs: ", tag_idxs)
            HT_DATA = X_data_scaled[tag_idxs, :]
            print("HT_DATA: ", HT_DATA)
            X_DATA_BY_HASHTAG.append(HT_DATA)


        # N_HT
        # unique_hashtags
        # X_DATA_BY_HASHTAG # list of numpy arrays for each ht (lat, long, time)
        print('X_DATA_BY_HASHTAG: ', X_DATA_BY_HASHTAG)


        # Get number of times each hashtag appears
        N_HT_INSTANCES = [len(array) for array in X_DATA_BY_HASHTAG]
        N_HT_INSTANCES = np.array(N_HT_INSTANCES)


        ################################################################################
        ################################################################################
        '''                         (1) PLOT SYNTHETIC DATA                          '''
        ################################################################################
        ################################################################################

        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.tri as mtri

        # Improve figure appearence
        import matplotlib as mpl
        mpl.rc('axes', labelsize=14)
        mpl.rc('xtick', labelsize=12)
        mpl.rc('ytick', labelsize=12)

        # enable saving figures
        import os
        PROJECT_PATH = "."
        FOLDER_NAME = "figures"
        IMAGES_PATH = os.path.join(PROJECT_PATH, "images", FOLDER_NAME)
        os.makedirs(IMAGES_PATH, exist_ok=True)

        def save_fig(fig_name, tight_layout=True, format="png", dpi=300):
            path = os.path.join(IMAGES_PATH, fig_name + "." + format)
            print("Saving figure", fig_name)
            if tight_layout:
                plt.tight_layout()
            plt.savefig(path, format=format, dpi=dpi)

        # plot synthetic data (2D, no time)
        figs_2d = []

        for ii in range(N_HT):
            fig_i = plt.figure(figsize=(6.4, 4.8))
            ax = fig_i.add_subplot(111)
            # plot scatter with alpha of 0.5
            HT_DATA = X_DATA_BY_HASHTAG[ii]
            ax.scatter(HT_DATA[:,0], HT_DATA[:,1], c='g', marker='.', label='HT '+str(ii)+':  #'+unique_hashtags[ii]+')')
            ax.set_title('Synthetic data in 2D (HT '+str(ii)+': #'+unique_hashtags[ii]+')')
            ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
            ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
            ax.legend(loc='upper right')
            save_fig('Synthetic data in 2D (HT '+str(ii)+' of '+str(N_HT)+')')
            plt.draw()
            figs_2d.append(fig_i)

        for ii in range(len(figs_2d)):
            figs_2d[ii].show()


        # plot synthetic data (3D, time)
        figs_3d = []
        for ii in range(N_HT):
            # plot scattering of data


            fig_i = plt.figure(figsize=(6.4, 4.8))
            ax = fig_i.add_subplot(111, projection='3d')
            HT_DATA = X_DATA_BY_HASHTAG[ii]
            ax.scatter(HT_DATA[:,0], HT_DATA[:,1], HT_DATA[:,2], c='g', marker='.', label='HT '+str(ii)+': #'+unique_hashtags[ii]+')')
            ax.set_title('Synthetic data in 3D (HT '+str(ii)+': #'+unique_hashtags[ii]+')')
            ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
            ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
            ax.set_zlabel('$x_3$ (time)', fontsize=18)
            ax.legend(loc='upper right')
            save_fig('Synthetic data in 3D (HT '+str(ii)+' of '+str(N_HT)+')')
            plt.draw()
            figs_3d.append(fig_i)

        for ii in range(len(figs_3d)):
            figs_3d[ii].show()

        # input(['Press enter to continue'])

        for ii in range(len(figs_2d)):
            plt.close(figs_2d[ii])

        for ii in range(len(figs_3d)):
            plt.close(figs_3d[ii])







        # N_HT
        # unique_hashtags
        # X_DATA_BY_HASHTAG # list of numpy arrays for each ht (lat, long, time)

        ################################################################################
        ################################################################################
        ''' (2) Generate PDFs of Xs (lattitude, longitude, time) across all hashtags '''
        '''
        i.e., generate
        P(x1,x2,x3|H1) =
        P(x1,x2,x3|H2) =
        etc.

        Later, will use Bayes to back-calculate
        P(H1|x1,x2,x3) = P(H1)*P(x1,x2,x3|H1)/P(x1,x2,x3)

        '''

        ################################################################################
        ################################################################################

        from sklearn.mixture import BayesianGaussianMixture
        from scipy.stats import multivariate_normal

        hashtag_mixture_models = []
        for ii in range(N_HT):
            Ndata = N_HT_INSTANCES[ii]
            initial_clusters = int(np.ceil(np.sqrt(Ndata))) # number of clusters to start BGMM with
            bgm = BayesianGaussianMixture(n_components=initial_clusters, n_init=10, max_iter = 200)
            bgm.fit(X_DATA_BY_HASHTAG[ii])
            hashtag_mixture_models.append(bgm)

        def pdf_gaussian_mixture(bgm, x):
            # Calculate the probability density function of a gaussian mixture model
            # for point or points x
            weights = bgm.weights_
            means = bgm.means_
            cov = bgm.covariances_

            n_components = len(weights)

            pdf = 0.
            for jj in range(n_components):
                rv = multivariate_normal(means[jj], cov[jj])
                gauss_pdf = rv.pdf(x)
                pdf += weights[jj] * gauss_pdf

            return(pdf)


        # save and load clustering info
        model_filename = "BGMM_models_agent_1.pkl"
        # save
        outfile = open(model_filename, 'wb')
        pickle.dump(hashtag_mixture_models, outfile)
        outfile.close()
        # load
        infile = open(model_filename, 'rb')
        hashtag_mixture_models = pickle.load(infile)
        infile.close()


        # save and load hashtag info
        hashtag_filename = "hashtag_info_agent_1.pkl"
        # save
        hashtag_info = { }
        hashtag_info["unique_hashtags"] = unique_hashtags
        hashtag_info["N_HT_INSTANCES"] = N_HT_INSTANCES
        outfile = open(hashtag_filename, 'wb')
        pickle.dump(hashtag_info, outfile)
        outfile.close()
        #load
        infile = open(hashtag_filename, 'rb')
        hashtag_info = pickle.load(infile)
        infile.close()
        unique_hashtags = hashtag_info["unique_hashtags"]
        N_HT_INSTANCES = hashtag_info["N_HT_INSTANCES"]
        N_HT = len(unique_hashtags)


        ################################################################################
        ################################################################################
        ''' (3) Find probability of each hashtag given data '''
        '''
        i.e., calculate P(H1|x1,x2,x3)

        Later, will use Bayes to back-calculate
        P(H1|x1,x2,x3) = P(H1)*P(x1,x2,x3|H1)/P(x1,x2,x3)

        P_ratio(H1|x1,x2,x3) = P(H1)*P(x1,x2,x3|H1)

        '''
        ################################################################################
        ################################################################################


        ################################################################################
        ''' Get prior probabilites of each hashtag '''

        # based on urn problem (how frequent observations have been):
        P_prior_hashtags = (N_HT_INSTANCES + 1)/(np.sum(N_HT_INSTANCES)+N_HT)
        print('probabilities sum: ', np.sum(P_prior_hashtags))


        ################################################################################
        ''' Get probabilites of each hashtag using Bayes '''


        def hashtag_probabilities_bayes(x_predict):

            if len(x_predict.shape) == 1: # 1D array
                N_pred = 1
            else:
                N_pred = x_predict.shape[0]

            P_data_given_hts = np.zeros([N_pred, N_HT]) # Npredictions by Nhashtags
            for ii in range(len(hashtag_mixture_models)):
                bgm = hashtag_mixture_models[ii]
                pdf = pdf_gaussian_mixture(bgm, x_predict)
                P_data_given_hts[:,ii] = pdf

            P_ratio = P_prior_hashtags * P_data_given_hts # Npredictions by Nhashtags

            divisor = np.sum(P_ratio, 1)
            divisor = divisor.reshape([-1, 1])
            P_posterior = P_ratio / divisor # Npredictions by Nhashtags
            return(P_posterior)


        ################################################################################
        ''' Get entire surfaces for probabilites of each hashtag '''

        lb = np.zeros(3)
        ub = np.ones(3)

        x1_plot = np.linspace(lb[0], ub[0], 65)
        x2_plot = np.linspace(lb[1], ub[1], 65)

        x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)
        #x3 = (lb[2]+ub[2])/2 * np.ones([x1_plot.size, 1])

        # plots at various times
        frac_times = [0, 0.5, 1]
        for frac_time in frac_times:

            x3 = frac_time * np.ones([x1_plot.size, 1])

            x_data = np.hstack((x1_plot.reshape([-1, 1]), x2_plot.reshape([-1, 1]), x3))

            P_posteriors = hashtag_probabilities_bayes(x_data)

            ################################################################################
            ''' plot each probability contour in 2D '''

            # plot probability contours for each hashtag

            # plot synthetic data (2D, no time)

            figs_probs = []

            for ii in range(N_HT):
                fig_i = plt.figure(figsize=(6.4, 4.8))
                ax = fig_i.add_subplot(111)

                # plot contour
                P_posterior = P_posteriors[:,ii]
                P_posterior = P_posterior.reshape(x1_plot.shape)
                c1 = ax.contourf(x1_plot, x2_plot, P_posterior, cmap='copper') #'copper','hot','RdYlGn','coolwarm','RdYlBu','RdGy',  #, color=HT_COLORS[ii], alpha = 0.5, )
                c2 = ax.contour(x1_plot, x2_plot, P_posterior, cmap='gray')#, color=HT_COLORS[ii], alpha = 0.5, )
                fig_i.colorbar(c1)

                # plot scatter with alpha of 0.5
                HT_DATA = X_DATA_BY_HASHTAG[ii]
                # tag_idxs = SYNTHETIC_DATA[:,0]
                # HT_DATA = SYNTHETIC_DATA[tag_idxs==ii+1, 1:]
                ax.scatter(HT_DATA[:,0], HT_DATA[:,1], c='g', marker='.', label='Probability contours HT '+str(ii)+': #'+unique_hashtags[ii])

                ax.set_title('probability contours (HT '+str(ii)+': #'+unique_hashtags[ii]+')')
                ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
                ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
                ax.legend(loc='upper right')
                save_fig('probability contours time '+str(frac_time)+' (HT '+str(ii)+')')
                plt.draw()
                figs_probs.append(fig_i)


            for ii in range(len(figs_probs)):
                figs_probs[ii].show()

            # input(['Press enter to continue'])

            for ii in range(len(figs_probs)):
                plt.close(figs_probs[ii])

        ################################################################################
        ''' Predict at requested point '''
        Xpredict = np.array([[lattitude, longitude, post_time]])
        Xpredict = xscale_obj.transform(Xpredict)

        probabilities = hashtag_probabilities_bayes(Xpredict)
        probabilities = probabilities[0]

        print("probabilities: ", probabilities)
        print("unique_hashtags: ", unique_hashtags)

        sort_idx = np.argsort(probabilities)[::-1]

        sorted_hashtags = unique_hashtags[sort_idx]
        sorted_probabilities = probabilities[sort_idx]
        print("sorted_probabilities: ", sorted_probabilities)
        print("unique_hashtags: ", unique_hashtags)

        return(sorted_hashtags, sorted_probabilities)


if __name__ == "__main__":
    hashtags, probabilities = RecoverHashtag(30.0, -80.0, 1.648e9)
    print("hashtags: ", hashtags)
    print("probabilities: ", probabilities)

#
