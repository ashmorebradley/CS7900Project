import numpy as np
from numpy.random import default_rng
################################################################################
################################################################################
'''                       (0) GENERATE SYNTHETIC DATA                        '''
################################################################################
################################################################################

N_HT = 5 # number of hashtags
N_CITY = 6 # number of cities (including countryside as city 0)
N_X = 3 # 3 variables

def generate_location_countryside(lb, ub, rng):
    # generate tweet location for the countryside
    x1 = rng.uniform(low=lb[0], high=ub[0])
    x2 = rng.uniform(low=lb[1], high=ub[1])
    X = np.array([x1, x2])
    return(X, rng)

def generate_location_city(mu, sigma, lb, ub, rng):
    # generate tweet location for a city
    # mu: mean locations (x1, x2)
    # sigma: standard deviation (s)
    X = np.array([])
    for ii in range(len(mu)):
        while True:
            x = mu[ii] + sigma * rng.standard_normal()
            if x >= lb[ii] and x <= ub[ii]:
                break
        X = np.append(X, x)
    return(X, rng)

def generate_time(mu_t, sigma_t, p_timeless, lb, ub, rng):
    # generate tweet time
    timeless = rng.choice([True, False], p=[p_timeless, 1-p_timeless])
    if timeless:
        # uniform sample
        t = rng.uniform(low=mu_t, high=ub)
    else:
        # normal sample
        while True:
            t = mu_t + sigma_t * rng.standard_normal()
            if t >= lb and t <= ub:
                break
    return(t, rng)

rng = default_rng()

lb = np.zeros(N_X)
ub = 100*np.ones(N_X)



# X, rng = generate_location_city(mu, sigma, lb, ub, rng)
# X, rng = generate_location_countryside(lb, ub, rng)
# t, rng = generate_time(mu_t, sigma_t, p_timeless, lb[2], ub[2], rng)

#                        Lattitude, longitude
CITY_CENTERS = np.array([[75, 30], #city 1
                         [20, 90], #city 2
                         [30, 20], #city 3
                         [25, 50], #city 4
                         [70, 80], #city 5
                         [95, 60]  #city 6
                        ])

CITY_RADII = np.array([15, 10, 8, 5, 5, 3])

CITY_PROB = CITY_RADII**1.5
# #                          city  1  2  3  4  5  6
# RELATIVE_POPULARITY = [np.array([9, 5, 5, 4, 6, 7]), # hashtag 1 (big city)
#                        np.array([6, 7, 7, 6, 5, 6]), # hashtag 2 (even, city)
#                        np.array([8, 2, 9, 5, 4, 5]), # hashtag 3 (south)
#                        np.array([2, 2, 5, 8, 8, 9]), # hashtag 4 (small)
#                        np.array([3, 4, 3, 3, 4, 5])  # hashtag 5 (countryside)
#                       ]
#
# COUNTRYSIDE_POPULARITY = [10, 2, 10, 10, 50]

#                          city  1  2  3  4  5  6
RELATIVE_POPULARITY = [np.array([9, 3, 2, 3, 1, 2]), # hashtag 1 (big city)
                       np.array([5, 7, 7, 6, 5, 6]), # hashtag 2 (even, city)
                       np.array([8, 2, 9, 5, 4, 5]), # hashtag 3 (south)
                       np.array([1, 1, 4, 8, 8, 9]), # hashtag 4 (small)
                       np.array([3, 4, 3, 3, 4, 5])  # hashtag 5 (countryside)
                      ]

COUNTRYSIDE_POPULARITY = [5, 2, 10, 5, 75]


CITY_PROBABILITIES = []
for ii in range(N_HT):
    probs = RELATIVE_POPULARITY[ii] * CITY_PROB
    cside_prob = COUNTRYSIDE_POPULARITY[ii]/100.
    probs = probs/np.sum(probs) * (1-cside_prob)
    probs = np.append(probs, cside_prob)
    CITY_PROBABILITIES.append(probs)

#              hashtag  1,  2,  3,  4,  5
TIME_PEAKS = np.array([10, 30, 30, 60, 80])
TIME_SIGS  = np.array([25, 30, 10, 20, 30])*2
P_TIMELESS = np.array([.05, .05, .05, .05, .05])


# N_HT_INSTANCES = np.array([50, 100, 30, 80, 60])
N_HT_INSTANCES = np.array([1000, 900, 700, 800, 600])

HT_COLORS = ['slategray','tab:blue','darkred','g','orange']

# Layout:
# [ht_idx, x1, x2, x3]
SYNTHETIC_DATA = np.array([])

# for each hashtag
for ii in range(N_HT):
    mu_t = TIME_PEAKS[ii]
    sigma_t = TIME_SIGS[ii]
    p_timeless = P_TIMELESS[ii]
    # for each hashtag instance
    for jj in range(N_HT_INSTANCES[ii]):
        # select city
        timeless = rng.choice([True, False], p=[p_timeless, 1-p_timeless])
        city_idx = rng.choice(N_CITY+1, p=CITY_PROBABILITIES[ii])
        # generate location
        if city_idx == N_CITY:
            L, rng = generate_location_countryside(lb, ub, rng)
        else:
            mu = CITY_CENTERS[city_idx,:]
            sigma = CITY_RADII[city_idx]
            L, rng = generate_location_city(mu, sigma, lb, ub, rng)
        # generate time
        t, rng = generate_time(mu_t, sigma_t, p_timeless, lb[2], ub[2], rng)
        # append data
        X = np.append(L, t)
        X = np.append(ii+1, X)

        if SYNTHETIC_DATA.size:
            SYNTHETIC_DATA = np.vstack([SYNTHETIC_DATA, X])
        else:
            SYNTHETIC_DATA = X

# separate out data by hashtag
SYNTHETIC_DATA_HASHTAGS = []
for ii in range(N_HT):
    tag_idxs = SYNTHETIC_DATA[:,0]
    HT_DATA = SYNTHETIC_DATA[tag_idxs==ii+1, 1:]
    SYNTHETIC_DATA_HASHTAGS.append(HT_DATA)

# break into training set and validation set



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
    # plot cities
    ax.scatter(CITY_CENTERS[:,0], CITY_CENTERS[:,1], s=10*CITY_RADII**2,  marker='o', facecolor='none', edgecolors = 'k', linewidths=3, label='cities') #c='k',
    # plot scatter with alpha of 0.5
    tag_idxs = SYNTHETIC_DATA[:,0]
    HT_DATA = SYNTHETIC_DATA[tag_idxs==ii+1, 1:]
    ax.scatter(HT_DATA[:,0], HT_DATA[:,1], c=HT_COLORS[ii], marker='.', label='HT '+str(ii+1))
    ax.set_title('Synthetic data in 2D (HT '+str(ii+1)+' of '+str(N_HT)+')')
    ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
    ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
    ax.legend(loc='upper right')
    save_fig('Synthetic data in 2D (HT '+str(ii+1)+' of '+str(N_HT)+')')
    plt.draw()
    figs_2d.append(fig_i)


for ii in range(len(figs_2d)):
    figs_2d[ii].show()

# plot all together in 2D

fig_2d = plt.figure(figsize=(6.4, 4.8))
ax = fig_2d.add_subplot(111)
# plot cities
ax.scatter(CITY_CENTERS[:,0], CITY_CENTERS[:,1], s=10*CITY_RADII**2,  marker='o', facecolor='none', edgecolors = 'k', linewidths=3, label='cities') #c='k',
for ii in range(N_HT):
    # plot scatter with alpha of 0.5
    tag_idxs = SYNTHETIC_DATA[:,0]
    HT_DATA = SYNTHETIC_DATA[tag_idxs==ii+1, 1:]
    ax.scatter(HT_DATA[:,0], HT_DATA[:,1], c=HT_COLORS[ii], marker='.', label='HT '+str(ii+1))
ax.set_title('Synthetic data in 2D')
ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
ax.legend(loc='upper right')
save_fig('Synthetic data in 2D')
plt.draw()

fig_2d.show()


# plot synthetic data (3D, time)
figs_3d = []
for ii in range(N_HT):
    # plot scattering of data


    fig_i = plt.figure(figsize=(6.4, 4.8))
    ax = fig_i.add_subplot(111, projection='3d')
    # # plot cities
    # ax.scatter(CITY_CENTERS[:,0], CITY_CENTERS[:,1], s=10*CITY_RADII**2,  marker='o', facecolor='none', edgecolors = 'k', linewidths=3, label='cities') #c='k',
    # plot scatter with alpha of 0.5
    tag_idxs = SYNTHETIC_DATA[:,0]
    HT_DATA = SYNTHETIC_DATA[tag_idxs==ii+1, 1:]
    ax.scatter(HT_DATA[:,0], HT_DATA[:,1], HT_DATA[:,2], c=HT_COLORS[ii], marker='.', label='HT '+str(ii+1))
    ax.set_title('Synthetic data in 3D (HT '+str(ii+1)+' of '+str(N_HT)+')')
    ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
    ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
    ax.set_zlabel('$x_3$ (time)', fontsize=18)
    ax.legend(loc='upper right')
    save_fig('Synthetic data in 3D (HT '+str(ii+1)+' of '+str(N_HT)+')')
    plt.draw()
    figs_3d.append(fig_i)

for ii in range(len(figs_3d)):
    figs_3d[ii].show()


# plot all together in 3D
fig_3d = plt.figure(figsize=(6.4, 4.8))
ax = fig_3d.add_subplot(111, projection='3d')
# # plot cities
# ax.scatter(CITY_CENTERS[:,0], CITY_CENTERS[:,1], s=10*CITY_RADII**2,  marker='o', facecolor='none', edgecolors = 'k', linewidths=3, label='cities') #c='k',
# plot scatter with alpha of 0.5
tag_idxs = SYNTHETIC_DATA[:,0]
for ii in range(N_HT):
    HT_DATA = SYNTHETIC_DATA[tag_idxs==ii+1, 1:]
    ax.scatter(HT_DATA[:,0], HT_DATA[:,1], HT_DATA[:,2], c=HT_COLORS[ii], marker='.', label='HT '+str(ii+1))
ax.set_title('Synthetic data in 3D')
ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
ax.set_zlabel('$x_3$ (time)', fontsize=18)
ax.legend(loc='upper right')
save_fig('Synthetic data in 3D')
plt.draw()

fig_3d.show()

'''
Bring back eventually
'''
# input(['Press enter to continue'])

for ii in range(len(figs_2d)):
    plt.close(figs_2d[ii])

plt.close(fig_2d)

for ii in range(len(figs_3d)):
    plt.close(figs_3d[ii])

plt.close(fig_3d)





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
    bgm = BayesianGaussianMixture(n_components=N_CITY+2, n_init=10, max_iter = 200)
    bgm.fit(SYNTHETIC_DATA_HASHTAGS[ii])
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

# single point to predict at
x_predict = np.array([50, 50, 50])

P_posterior = hashtag_probabilities_bayes(x_predict)

print('P_posterior: ', P_posterior)


################################################################################
''' Get entire surfaces for probabilites of each hashtag '''


x1_plot = np.linspace(lb[0], ub[0], 65)
x2_plot = np.linspace(lb[1], ub[1], 65)

x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)
x3 = (lb[2]+ub[2])/2 * np.ones([x1_plot.size, 1])

x_data = np.hstack((x1_plot.reshape([-1, 1]), x2_plot.reshape([-1, 1]), x3))

P_posteriors = hashtag_probabilities_bayes(x_data)

################################################################################
''' plot all probability surfaces in 3D '''
fig_probabilities = plt.figure(figsize=(6.4, 4.8))
ax = fig_probabilities.add_subplot(111, projection='3d')
#ax.rcParams['legend.fontsize'] = 10
# plot cities
ax.scatter(CITY_CENTERS[:,0], CITY_CENTERS[:,1], s=10*CITY_RADII**2,  marker='o', facecolor='none', edgecolors = 'k', linewidths=3, label='cities') #c='k',
# plot surfaces with alpha of 0.5
tag_idxs = SYNTHETIC_DATA[:,0]
for ii in range(N_HT):
    P_posterior = P_posteriors[:,ii]
    P_posterior = P_posterior.reshape(x1_plot.shape)
    surf = ax.plot_surface(x1_plot, x2_plot, P_posterior, color=HT_COLORS[ii], alpha = 0.5, label='HT '+str(ii+1))
    surf._facecolors2d=surf._facecolor3d
    surf._edgecolors2d=surf._edgecolor3d
    # HT_DATA = SYNTHETIC_DATA[tag_idxs==ii+1, 1:]
    # ax.scatter(HT_DATA[:,0], HT_DATA[:,1], HT_DATA[:,2], c=HT_COLORS[ii], marker='.', label='HT '+str(ii+1))
ax.set_title('Hashtag Probabilities (time = 50)')
ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
ax.set_zlabel('$probability$', fontsize=18)
ax.legend()#loc='lower right')
save_fig('hashtag_probabilities_each_location')
plt.draw()

fig_probabilities.show()



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
    c1 = ax.contourf(x1_plot, x2_plot, P_posterior, label='Probability contours HT '+str(ii+1), cmap='copper')#'copper','hot','RdYlGn','coolwarm','RdYlBu','RdGy',  #, color=HT_COLORS[ii], alpha = 0.5, )
    c2 = ax.contour(x1_plot, x2_plot, P_posterior, label='Probability contours HT '+str(ii+1), cmap='gray')#, color=HT_COLORS[ii], alpha = 0.5, )
    fig_i.colorbar(c1)

    # plot cities
    ax.scatter(CITY_CENTERS[:,0], CITY_CENTERS[:,1], s=10*CITY_RADII**2,  marker='o', facecolor='none', edgecolors = 'k', linewidths=3, label='cities') #c='k',
    # plot scatter with alpha of 0.5
    tag_idxs = SYNTHETIC_DATA[:,0]
    HT_DATA = SYNTHETIC_DATA[tag_idxs==ii+1, 1:]
    ax.scatter(HT_DATA[:,0], HT_DATA[:,1], c=HT_COLORS[ii], marker='.', label='HT '+str(ii+1))

    ax.set_title('probability contours (HT '+str(ii+1)+' of '+str(N_HT)+')')
    ax.set_xlabel('$x_1$ (latitude)', fontsize=18)
    ax.set_ylabel('$x_2$ (longitude)', fontsize=18)
    ax.legend(loc='upper right')
    save_fig('probability contours (HT '+str(ii+1)+')')
    plt.draw()
    figs_probs.append(fig_i)


for ii in range(len(figs_probs)):
    figs_probs[ii].show()




input(['Press enter to continue'])

plt.close(fig_probabilities)

for ii in range(len(figs_probs)):
    plt.close(figs_probs[ii])









#
