import numpy as np
import random
import graph_search as gs

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
    features1 = normalize(features1)
    features2 = normalize(features2)

    distance_arr = euclidean_distance(features1, features2)
    sorted_distance_idxs = np.argsort(distance_arr, axis=1)

    # frontier = gs.get_frontier(distance_arr)
    # match_search = gs.graph_search(distance_arr, frontier)
    # import pdb; pdb.set_trace()

    # get all possible match candidates (fv1-fv2 pair determined by finding fv2 idx of min distance for each feature in fv1)
    # less than ideal bc the available min for fv1 is not necessarily optimal
    available_features2 = np.arange(len(features2))
    match_candidates = np.empty((0,2), dtype=int)

    # for each sorted fv1 idx row
    for fv1_idx, idx_row in enumerate(sorted_distance_idxs):
        # for each fv2 value in idx
        for fv2_idx in idx_row:
            if fv2_idx in available_features2:
                match_candidates = np.append(match_candidates, np.array([[fv1_idx, fv2_idx]]), axis=0)
                matched_index = np.where(available_features2 == fv2_idx)
                available_features2 = np.delete(available_features2, matched_index)
                break

    # calculate confidence of each match candidate
    confidence_candidates = np.array([], dtype=float)
    sorted_distance_arr = np.array([distance_arr[i, idxs] for i, idxs in enumerate(sorted_distance_idxs)])

    # calculate nndr as nn1 / nn2
    nn1_arr = sorted_distance_arr[:,:-1]
    nn2_arr = sorted_distance_arr[:,1:]
    confidences_arr = nn1_arr / nn2_arr
    confidences_t = np.transpose(confidences_arr)

    # get the min nndr for both fv1 and fv2 sets
    fv1_mins = np.array([np.min(row) for row in confidences_arr])
    fv2_mins = np.array([np.min(row) for row in confidences_t])

    # TODO: check feature lengths so not out of idx
    neighbor_arr = np.empty((len(fv1_mins), len(fv2_mins)), dtype=float)

    for i in np.arange(len(fv1_mins)):
        for j in np.arange(len(fv2_mins)):
            neighbor_arr[i][j] = fv1_mins[i] - fv2_mins[j]

    neighbor_arr = np.sqrt(np.square(neighbor_arr))

    for fv1_idx, row in enumerate(neighbor_arr):
        row_min = np.min(row)
        for fv2_idx, val in enumerate(row):
            if row_min == val:
                nndr = confidences_arr[fv1_idx][fv2_idx]
                confidence_candidates = np.append(confidence_candidates, np.array([nndr]), axis=0)
                break

    top_indices = get_top_sorted_indices(confidence_candidates, top=120)

    return match_candidates[top_indices,], confidence_candidates[top_indices,]

def normalize(arr):
    """
        Calculates normal as sqrt(sum of sqrs)

        Args:
        - arr: A numpy array of unknown shape (could either be feature patch or feature vector)
        Returns:
        - normalized arr: The normalized arr numpy array
    """
    ndims = len(arr.shape)
    if ndims == 2:
        return arr / np.linalg.norm(arr)
    else:
        return np.array([normalize(patch) for patch in arr])

def euclidean_distance(arr1, arr2):
    """
        Calculates euclidean distance between two n-dimensional arrays (sqrt of sum of squared difference)

        Args:
        - arr1: numpy array of unknown shape (could either be feature patch or feature vector)
        - arr2: numpy array of similarly unknown shape as arr1 (could either be feature patch or feature vector)
        Returns:
        - distances_arr: n-dimensional numpy array of scalar euclidean distances
    """

    assert arr1.shape == arr2.shape or len(arr1.shape) > 2, 'euclidean distance must be performed on similarly shaped arrays'

    ndims = len(arr1.shape)
    if ndims == 2:
        sqr = np.square(arr1 - arr2)
        total = np.sum(sqr)
        return np.sqrt(total)
    else:
        arr1_feature_count = arr1.shape[0]
        arr2_feature_count = arr2.shape[0]
        distances_arr = np.empty([arr1_feature_count, arr2_feature_count], float)

        for patch_index, arr1_patch in enumerate(arr1):
            distances_arr[patch_index] =  np.array([euclidean_distance(arr1_patch, arr2_patch) for arr2_patch in arr2])
        return distances_arr

def nndr_order(nndr_arr):
    ordered_rows = [np.argsort(nndrs) for nndrs in nndr_arr[:,:,2]]
    return np.dstack((nndr_arr, ordered_rows))

def add_nndrs(ordered_arr):
    # calculates nndrs and adds to depth stack of ordered_arr
    distance_rows = ordered_arr[:,:,0]
    fwd_distances = [np.append(row[1:], 1) for row in distance_rows]
    nndrs = distance_rows / fwd_distances
    nndrs = nndrs[:,:,np.newaxis]
    return np.dstack((ordered_arr, nndrs))

def order_by_distance(distance_arr):
    # adds current distance indices to stack and returns array sorted by distance
    arr_indices = np.indices(distance_arr.shape)
    col_indices = arr_indices[1]
    distance_idx_arr = np.dstack((distance_arr, col_indices))
    sorted_by_distance = np.sort(distance_idx_arr, axis=1)

    return sorted_by_distance

def distance_ordering(distance_arr):
    # returns MxNx2 stacked numpy array with distance ranks at second level depth
    distance_indices = np.array([np.argsort(row) for row in distance_arr])
    return np.dstack((distance_arr, distance_indices))

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
