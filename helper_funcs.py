"""
Helper functions for preparing, filtering and displaying the disparity maps.
"""
import numpy as np
from matplotlib import cm


def filter_disparity(disp, conf, threshold):
    """
    Filters out all disparities with confidence
    lower than the threshold value.
    Then returns the filtered disparity map.

    Arguments:
        - disp: disparity map with dimensions H x W.
        - conf: confidence map with dimensions H x W.
        - threshold: minimum confidence required to keep 
            the disparity value.

    Returns: Filtered disparity map with dimensions H x W.
    """
    filtered_conf = np.where(conf >= threshold, disp, 0)
    return filtered_conf


def select_disparity(aggregation_volume):
    """
    Converts the aggregation volume into a disparity map using 
    the winner takes all strategy. 
    Cost volume is first calculated by taking the sum of the costs over all paths.
    Then the disparities are determined by finding the 
    disparity index with the lowest cost for the pixel.

    Arguments:
        - aggregation_volume: Array containing the matching costs for 
            all pixels at all disparities and paths, with dimension H x W x D x N

    Returns: Disparity map with dimensions H x W.
    """
    # sum up costs for all directions
    volume = np.sum(aggregation_volume, axis=3) 
    # returns the disparity index with the minimum cost associated with each h x w pixel
    disparity_map = np.argmin(volume, axis=2) 
    return disparity_map


def normalize(disp, max_disparity):
    """
    Normalizes the disparity map, then
    quantizes it so that it can be displayed. 
    Arguments:
        - disp: disparity map with dimensions H x W.
        - max_disparity: maximum disparity of the array.
    
    Return: normalized then quantized array, ready for visualization.
    """
    return 255.0 * disp / max_disparity


def colorize_image(image, cmap='jet'):
    """
    Converts single channel matrix with quantized values
    to an RGB colorized version.
    Arguments:
      - image: Quantized (uint8) single channel image with dimensions H x W 
      - cmap: a valid cmap named for use with matplotlib's 'get_cmap'
    
    Returns an RGB depth map with dimension H x W x 3.
    """
    color_map = cm.get_cmap(cmap)
    colors = color_map(np.arange(256))[:, :3].astype(np.float32)
    colorized_map = np.take(colors, image, axis=0)
    colorized_map = np.uint8(colorized_map * 255)
    return colorized_map