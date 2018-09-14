import numpy as np
import random

def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    matches = random_features(features1, features2)
    confidences = np.empty([0,1])


    # calculate euclidean distance of all 16x16 dimensions of each feature with each other feature
    # find ratio of feature's distance to nearest and next nearest neighbor (confidence)
    # return only the top 100 most confident matches

    return matches, confidences

def random_features(features1, features2):
    max_k = min(len(features1), len(features2))
    k = random.randrange(1, max_k)

    f1_indices = range(len(features1))
    f2_indices = range(len(features2))

    random1 = random.sample(f1_indices, k)
    random2 = random.sample(f2_indices, k)

    # create array, transpose, concatenate
    rando_arr = np.array([random1, random2])
    rando_arr = np.transpose(rando_arr)

    assert rando_arr.shape == (k, 2), 'random feature matches must be shape (k, 2)'

    return rando_arr

def feature_distances(features):
    dimensionality = len(features)
    distances = np.zeros((dimensionality, dimensionality))

    rows = [get_distance_row(features, dim, dimensionality) for dim in np.arange(dimensionality-1)]
    upper_triangle = np.array(rows)

    # reflect upper triangle along diagonal
    # transpose upper to get lower
    lower_triangle = np.transpose(upper_triangle)

    # zero out the diagonal to prevent double counting
    lower_triangle = np.tril(lower_triangle, k=-1)

    # add to get symmetrical distance matrix
    return upper_triangle + lower_triangle


def get_distance_row(features, dim, dimensionality):
    # returns (k,) array with leading zeros
    # get fwd range
    fwd_indices = np.arange(dim + 1, dimensionality)

    # leading zeros
    entry_distances = np.zeros(dimension)

    # get distance from current dimension for each fwd index
    current_feature = features[dim]
    fwd_distances = [euclidean_distance(current_feature, features[fwd_index]) for fwd_index in fwd_indices]

    return np.concatenate((entry_distances, fwd_distances), axis=None)

def euclidean_distance(fv1, fv2):
    # sum of element wise square root of squared difference
    # returns floating point scalar
    sqr = np.square(fv1 - fv2)
    sqrt = np.sqrt(sqr)
    return np.sum(sqrt)
