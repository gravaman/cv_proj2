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

    coordinates = xy_coordinates(x, y)
    bounded_coordinates = remove_edge_points(image, coordinates, feature_width)

    fvs = get_local_patches(image, bounded_coordinates, feature_width)

    return fvs

def remove_edge_points(image, coordinates, feature_width):
    x_coordinates, y_coordinates = coordinates
    [x_min, x_max], [y_min, y_max] = image_bounds(image, feature_width)

    bounded_coords = [(x_val, y_val) for x_val, y_val in zip(x_coordinates, y_coordinates) if (x_min <= x_val <= x_max) and (y_min <= y_val <= y_max)]
    x_coords = [x_val for x_val, _ in bounded_coords]
    y_coords = [y_val for _, y_val in bounded_coords]
    return (x_coords, y_coords)

def image_bounds(image, feature_width):
    # takes np image and returns [x_min, x_max], [y_min, y_max]
    image_y_max, image_x_max = image.shape
    return [feature_width, image_x_max - feature_width], [feature_width, image_y_max - feature_width]

def pad_image(image, feature_width):
    # returns padded image copy allowing feature detection near edges
    pad_size = get_pad_size(feature_width)

    padded_image = np.pad(image, pad_size, 'symmetric')
    assert padded_image.shape == (image.shape[0] + feature_width, image.shape[1] + feature_width)

    return padded_image

def get_pad_size(feature_width):
    assert feature_width % 2 == 0, 'feature width must be even'
    return math.trunc(feature_width / 2)

def xy_coordinates(x_vals, y_vals, pad_size=0):
    # buffer values and return (x, y)
    assert len(x_vals) == len(y_vals), 'y and x location values must be balanced'

    flat_x_vals = x_vals.flatten()
    flat_y_vals = y_vals.flatten()

    # pad_size = get_pad_size(feature_width)
    x_vals_round = [int(round(x_val)) for x_val in flat_x_vals]
    y_vals_round = [int(round(y_val)) for y_val in flat_y_vals]

    return (x_vals_round, y_vals_round)

def get_local_patches(padded_image, pixel_coordinates, feature_width):
    patches = []
    x_coords, y_coords = pixel_coordinates
    for x_val, y_val in zip(x_coords, y_coords):
        (y_entry, y_exit), (x_entry, x_exit) = get_patch_bounds(y_val, x_val, feature_width) # numpy row, col order
        patch = padded_image[y_entry:y_exit, x_entry:x_exit] # numpy row, col order
        normal_patch = normalize_patch(patch)

        assert normal_patch.shape == (16,16), 'patch shape is not 16x16'
        patches.append(normal_patch)

    return np.array(patches)

def normalize_patch(patch):
    magnitude = np.linalg.norm(patch)
    return patch / magnitude

def get_patch_bounds(y_val, x_val, feature_width):
    # returns in numpy row, col order
    assert feature_width % 2 == 0, 'feature width must be even'
    entry_buffer = int(feature_width / 2 - 1)
    exit_buffer = int(feature_width / 2 + 1) # need to add bc not inclusive

    y_bounds = (y_val - entry_buffer, y_val + exit_buffer)
    x_bounds = (x_val - entry_buffer, x_val + exit_buffer)

    return y_bounds, x_bounds
