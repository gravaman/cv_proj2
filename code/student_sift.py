import numpy as np
import cv2
import math

def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    padded_image = pad_image(image, feature_width)
    coordinates = yx_coordinates(y, x, feature_width)
    patches = get_local_patches(padded_image, coordinates, feature_width)

    fv = 'baller'
    return fv

def pad_image(image, feature_width):
    # returns padded image copy allowing feature detection near edges
    pad_size = get_pad_size(feature_width)

    padded_image = np.pad(image, pad_size, 'symmetric')
    assert padded_image.shape == (image.shape[0] + feature_width, image.shape[1] + feature_width)

    return padded_image

def get_pad_size(feature_width):
    assert feature_width % 2 == 0, 'feature width must be even'
    return math.trunc(feature_width / 2)

def get_local_patches(padded_image, pixel_coordinates, feature_width):
    patches = []
    for (y_val, x_val) in pixel_coordinates:
        (y_entry, y_exit), (x_entry, x_exit) = get_patch_bounds(y_val, x_val, feature_width)
        patch = padded_image[y_entry:y_exit, x_entry:x_exit]
        patches.append(patch)
    return np.array(patches)

def get_patch_bounds(y_val, x_val, feature_width):
    assert feature_width % 2 == 0, 'feature width must be even'
    entry_buffer = int(feature_width / 2 - 1)
    exit_buffer = int(feature_width / 2 + 1) # need to add bc not inclusive

    y_bounds = (y_val - entry_buffer, y_val + exit_buffer)
    x_bounds = (x_val - entry_buffer, x_val + exit_buffer)
    return y_bounds, x_bounds

def yx_coordinates(y_vals, x_vals, feature_width):
    # buffer values and return (y, x)
    assert len(y_vals) == len(x_vals), 'y and x location values must be balanced'

    flat_y_vals = y_vals.flatten()
    flat_x_vals = x_vals.flatten()

    pad_size = get_pad_size(feature_width)
    return [(int(round(flat_y_vals[index]) + pad_size), int(round(x_val) + pad_size)) for index, x_val in enumerate(flat_x_vals)]
