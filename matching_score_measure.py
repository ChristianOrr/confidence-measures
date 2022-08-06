"""
This is the python script version of the  
confidence estimation technique.
Very little explanations are provided for the methods in this script.
To see a detailed overview of this implementation see the jupyter
notebook with the same name.
"""

import argparse
import time as t
import cv2
import numpy as np
from matplotlib import cm
from sgm_numpy import sgm



def matching_score_measure(aggregation_volume, csize):
    """
    Creates a confidence map (H x W) using the 
    matching score measure technique from the 
    aggregation volume (H x W x D x N).

    Arguments:
        - aggregation_volume: Array containing the matching costs for 
            all pixels at all disparities and paths, with dimension H x W x D x N.
        - csize: kernel size for the census transform.

    Returns: Confidence map with normalized values (0-1) and 
        dimensions H x W.
    """
    # sum up costs for all directions
    volume = np.sum(aggregation_volume, axis=3, dtype=np.int32)     
    # returns the minimum cost associated with each h x w pixel
    min_costs = np.min(volume, axis=2) 
    # Negate the cost so that lower is less confidant and higher is more confident
    min_costs = -min_costs
    # Normalize to get confidence values between 0-1
    minimum = np.min(min_costs)
    maximum = np.max(min_costs)
    confidence = (min_costs - minimum) / (maximum - minimum)
    # Set border pixels confidence to 0 (since disparity couldn't be calculated there)
    confidence[:csize[0], :] = 0 # top border
    confidence[-csize[0]:, :] = 0 # bottom border
    confidence[:, :csize[1]] = 0 # left border
    confidence[:, -csize[1]:] = 0 # right border
    return confidence


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


def calc_roc(disp, gt, conf, tau):
    """
    Sorts the disparities based on decreasing confidence.
    Then calculates the errors for the decreasing confidence
    disparities. The mean errors at cumulitive intervals are then 
    added to an array to be used for the y-values in the ROC curve.

    The optimal ROC values are then calculated using the same method,
    but the errors are arranged by adding incorrect values 
    to the back of the array.

    Arguments:
        - disp: disparity map with dimensions H x W.
        - disp: groundtruth disparity map with dimensions H x W.
        - conf: confidence map with dimensions H x W.
        - tau: threshold used for D1 error rate, number of disparities with difference > tau.

    Returns: ROC curve, optimal ROC curve, 
        mean error sorted confidence array and 
        ratios of pixels used for x_axis in ROC curves
    """
    # Don't use pixels lacking ground truth disparity
    valid = gt > 0
    disp = disp[valid]
    gt = gt[valid]
    conf = conf[valid]

    # Sort the disparities based on decreasing order of confidence
    sorted_indices = np.argsort(conf)[::-1]
    sorted_disp = np.take_along_axis(disp, sorted_indices, axis=0)
    sorted_gt = np.take_along_axis(gt, sorted_indices, axis=0)
    sorted_conf = np.take_along_axis(conf, sorted_indices, axis=0)  

    # Get the errors (disparities with greater absolute difference than tau)
    errors = np.where(np.abs(sorted_disp - sorted_gt) > tau, 1, 0) 
    # Get the mean error rate on all pixels
    mean_error = np.sum(errors) / errors.shape[0]

    # Find the cumulitive ratios of data to calculate errors for the roc curve
    step_size = 0.05
    cumulitive_ratio = 0
    ratio_pixels = []
    while cumulitive_ratio < 1.0:
        cumulitive_ratio += step_size
        ratio_pixels.append(cumulitive_ratio)  

    # Calculate errors for roc curve
    roc = []
    for ratio in ratio_pixels:
        index = int(errors.shape[0] * ratio)
        number_of_errors = np.sum(errors[:index])
        error_percentage = number_of_errors / index
        roc.append(error_percentage)

    # Calculate optimal roc curve by placing all errors at the end of the array
    sorted_errors = np.sort(errors)
    optimal_roc = []
    for ratio in ratio_pixels:
        index = int(sorted_errors.shape[0] * ratio)
        number_of_errors = np.sum(sorted_errors[:index])
        error_percentage = number_of_errors / index
        optimal_roc.append(error_percentage)

    return roc, optimal_roc, mean_error, sorted_conf, ratio_pixels


def calc_auc(roc, optimal_roc):
    """
    Calculates the area under the ROC curve (AUC)
    for both the ROC and optimal ROC curves.

    Arguments:
        - roc: ROC curve y-values. 
            Errors sorted by confidence, then mean errors are 
            calculated at cumulitive disparities.
        - optimal_roc: optimal ROC curve y-values. 
            Incorrect values are move to the back of 
            errors array, then mean errors are calculated at 
            cumulitive disparities.

    Returns: AUC and optimal AUC values.
    """
    auc = np.trapz(roc, dx=1./20)
    optimal_auc = np.trapz(optimal_roc, dx=1./20)
    return auc, optimal_auc



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='teddy/im2.png', help='name (path) to the left image')
    parser.add_argument('--right', default='teddy/im6.png', help='name (path) to the right image')
    parser.add_argument('--left_gt', default='teddy/disp2.png', help='name (path) to the left ground-truth image')
    parser.add_argument('--right_gt', default='teddy/disp6.png', help='name (path) to the right ground-truth image')
    parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
    parser.add_argument('--disp', default=64, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--images', default=False, type=bool, help='save intermediate representations')
    parser.add_argument('--eval', default=True, type=bool, help='evaluate disparity map with 3 pixel error')
    parser.add_argument('--p1', default=10, type=int, help='penalty for disparity difference = 1')
    parser.add_argument('--p2', default=120, type=int, help='penalty for disparity difference > 1')
    parser.add_argument('--csize', default=[5, 5], nargs="+", type=int, help='size of the kernel for the census transform')
    parser.add_argument('--bsize', default=[3, 3], nargs="+", type=int, help='size of the kernel for blurring the images and median filtering')
    parser.add_argument('--tau', default=3, type=int, help='Used for D1 error rate: number of disparities with difference > tau')
    args = parser.parse_args()

    left_name = args.left
    right_name = args.right
    left_gt_name = args.left_gt
    right_gt_name = args.right_gt
    output_name = args.output
    save_images = args.images
    evaluation = args.eval
    max_disparity = args.disp
    P1 = args.p1
    P2 = args.p2
    csize = args.csize
    bsize = args.bsize
    tau = args.tau

    dawn = t.time()

    print('\nLoading images...')
    left = cv2.imread(left_name, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_name, cv2.IMREAD_GRAYSCALE)
    height = left.shape[0]
    width = left.shape[1]
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert max_disparity > 0, 'maximum disparity must be greater than 0.'

    print("\nPerforming SGM...")
    left_aggregation_volume, right_aggregation_volume = sgm(left, right, max_disparity, P1, P2, csize, bsize, height, width)


    print("\nCalculating confidences...")
    left_confidences = matching_score_measure(left_aggregation_volume, csize)
    right_confidences = matching_score_measure(right_aggregation_volume, csize)


    print('\nSelecting best disparities...')
    left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume), max_disparity))
    right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), max_disparity))


    print('\nApplying median filter...')
    left_disparity_map = cv2.medianBlur(left_disparity_map, bsize[0])
    right_disparity_map = cv2.medianBlur(right_disparity_map, bsize[0])



    if save_images:
        cv2.imwrite(f'left_{output_name}', colorize_image(left_disparity_map, "jet"))
        cv2.imwrite(f'right_{output_name}', colorize_image(right_disparity_map, "jet"))
        cv2.imwrite(f'left_conf_{output_name}', colorize_image(np.uint8(left_confidences * 255), "winter"))
        cv2.imwrite(f'right_conf_{output_name}', colorize_image(np.uint8(right_confidences * 255), "winter"))

    if evaluation:
        left_gt = cv2.imread(left_gt_name, cv2.IMREAD_GRAYSCALE)
        right_gt = cv2.imread(right_gt_name, cv2.IMREAD_GRAYSCALE)
        # Calculate ROC/optimal ROC curves
        left_roc, left_optimal_roc, left_mean_error, left_sorted_conf, ratio_pixels = calc_roc(left_disparity_map, left_gt, left_confidences, tau)
        right_roc, right_optimal_roc, right_mean_error, right_sorted_conf, ratio_pixels = calc_roc(right_disparity_map, right_gt, right_confidences, tau)
        # Calculate AUC/optimal AUC
        left_auc, left_optimal_auc = calc_auc(left_roc, left_optimal_roc)
        right_auc, right_optimal_auc = calc_auc(right_roc, right_optimal_roc)

        print("\nLeft Results: ")
        print(f"\tTotal Error: {left_mean_error:.3f}, AUC: {left_auc:.3f}, Optimal AUC: {left_optimal_auc:.3f}")
        print("\nRight Results: ")
        print(f"\tTotal Error: {right_mean_error:.3f}, AUC: {right_auc:.3f}, Optimal AUC: {right_optimal_auc:.3f}")

    dusk = t.time()
    print('\nFinished')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))