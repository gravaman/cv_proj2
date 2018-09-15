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

    distance_arr = feature_distances(features1, features2)
    match_candidates, confidence_candidates = match_along_row(distance_arr)
    top_indices = get_top_sorted_indices(confidence_candidates, top=100)

    return match_candidates[top_indices,], confidence_candidates[top_indices,]

def match_along_row(distance_arr, axis=0):
    # returns match locations (int axis=0, int axis=1) and confidences (float) across rows
    # row_dims, _ = distance_arr.shape
    matches = [] # np.empty([row_dims, 2], int)
    confidences = [] # np.empty([row_dims, 1], float)

    for i, row in enumerate(distance_arr):
        rankings = np.argsort(row)
        nn1_index, nn2_index = rankings[0:2]

        if axis == 1:
            fv1_index = nn1_index
            fv2_index = i
        else:
            fv1_index = i
            fv2_index = nn1_index

        matches.append([fv1_index, fv2_index])
        confidences.append(row[nn1_index] / row[nn2_index])

    return np.array(matches, dtype=int), np.array(confidences, dtype=float)

def get_top_sorted_indices(confidence_candidates, top=100):
    sorted_indices = np.argsort(confidence_candidates)
    return sorted_indices[:top]

def feature_distances(fv1, fv2):
    # (fv1_k x fv2_k) array of euclidean distances
    rows = [get_distances_for_patch(fv1_patch, fv2) for fv1_patch in fv1]
    return np.array(rows, dtype=float)

def get_distances_for_patch(fv1_patch, fv2):
    # returns (,k) array of euclidean distances between given fv1_patch and each fv2 patch
    return [euclidean_distance(fv1_patch, fv2_patch) for fv2_patch in fv2]

def euclidean_distance(fv1, fv2):
    # sum of element wise square root of squared difference
    # returns floating point scalar
    sqr = np.square(fv1 - fv2)
    sqrt = np.sqrt(sqr)
    return np.sum(sqrt)

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
