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
from scipy.signal import argrelextrema
from sgm_numpy import sgm
from evaluation_funcs import calc_roc, calc_auc
from helper_funcs import filter_disparity, select_disparity, normalize, colorize_image


def find_local_minima(volume, height, width):
    """
    Finds the two smallest local minima in the 
    cost volume for all pixels.

    Arguments:
        - volume: Array containing the matching costs for 
            all pixels at all disparities, with dimension H x W x D.
        - height: number of rows of the image.
        - width: number of columns of the image.
        
    Returns: The global mimimum costs, c1, for all pixels with dimension H x W, 
        and the second local minimum costs, c2m, for all pixels with dimension H x W.
    """

    c1 = np.zeros(shape=(height, width), dtype=np.int32)
    c2m = np.zeros(shape=(height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            minimum = np.min(volume[y, x, :])
            maximum = np.max(volume[y, x, :])
            # Get local minima indices
            local_min_disparities = argrelextrema(volume[y, x, :], np.less)[0]
            # Get local minima values
            local_min_costs = np.take(volume[y, x, :], local_min_disparities)
            local_min_costs = np.sort(local_min_costs)
            # If there of no troughs, then the values are either flat or monotonic. 
            # In either case we can take c1 to be the min cost and c2m as the max.
            if local_min_costs.size < 1:
                c1[y, x] = minimum
                c2m[y, x] = maximum
            # If there is only one trough, then the trough is either the local min or global min.
            # If the trough cost is equal to the min of all values, 
            # then the trough is the global min, otherwise its a local min.
            elif local_min_costs.size < 2:
                if local_min_costs[0] == minimum:
                    c1[y, x] = local_min_costs[0]
                    c2m[y, x] = maximum
                else:
                    c1[y, x] = minimum
                    c2m[y, x] = local_min_costs[0]
            # Ideal case when we have two or more troughs, then we can just take the smallest two troughs
            else:
                c1[y, x] = local_min_costs[0]
                c2m[y, x] = local_min_costs[1]
    return c1, c2m


def maximum_margin(aggregation_volume, csize, height, width):
    """
    Creates a confidence map (H x W) using the 
    maximum margin technique from the 
    aggregation volume (H x W x D x N).

    Arguments:
        - aggregation_volume: Array containing the matching costs for 
            all pixels at all disparities and paths, with dimension H x W x D x N.
        - csize: kernel size for the census transform.
        - height: number of rows of the image.
        - width: number of columns of the image.
        
    Returns: Confidence map with normalized values (0-1) and 
        dimensions H x W.
    """
    # sum up costs for all directions
    volume = np.sum(aggregation_volume, axis=3, dtype=np.int32) 
    # Find local minima
    c1, c2m = find_local_minima(volume, height, width)
    # Calculate MM
    MM = c2m - c1
    # Normalize to get confidence values between 0-1
    minimum = np.min(MM)
    maximum = np.max(MM)
    confidence = (MM - minimum) / (maximum - minimum)
    # Set border pixels confidence to 0 (since disparity couldn't be calculated there)
    confidence[:csize[0], :] = 0 # top border
    confidence[-csize[0]:, :] = 0 # bottom border
    confidence[:, :csize[1]] = 0 # left border
    confidence[:, -csize[1]:] = 0 # right border
    return confidence



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='teddy/im2.png', help='name (path) to the left image')
    parser.add_argument('--right', default='teddy/im6.png', help='name (path) to the right image')
    parser.add_argument('--left_gt', default='teddy/disp2.png', help='name (path) to the left ground-truth image')
    parser.add_argument('--right_gt', default='teddy/disp6.png', help='name (path) to the right ground-truth image')
    parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
    parser.add_argument('--disp', default=64, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--save', default=False, type=bool, help='save disparity and confidence maps')
    parser.add_argument('--eval', default=True, type=bool, help='evaluate disparity map with 3 pixel error')
    parser.add_argument('--p1', default=10, type=int, help='penalty for disparity difference = 1')
    parser.add_argument('--p2', default=120, type=int, help='penalty for disparity difference > 1')
    parser.add_argument('--csize', default=[5, 5], nargs="+", type=int, help='size of the kernel for the census transform')
    parser.add_argument('--bsize', default=[3, 3], nargs="+", type=int, help='size of the kernel for blurring the images and median filtering')
    parser.add_argument('--tau', default=3, type=int, help='Used for D1 error rate: number of disparities with difference > tau')
    parser.add_argument('--conf_threshold', default=0.7, type=float, help='Retain disparitys with confidence greater than the conf_threshold')
    args = parser.parse_args()

    left_name = args.left
    right_name = args.right
    left_gt_name = args.left_gt
    right_gt_name = args.right_gt
    output_name = args.output
    save_images = args.save
    evaluation = args.eval
    max_disparity = args.disp
    P1 = args.p1
    P2 = args.p2
    csize = args.csize
    bsize = args.bsize
    tau = args.tau
    conf_threshold = args.conf_threshold

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
    left_confidences = maximum_margin(left_aggregation_volume, csize, height, width)
    right_confidences = maximum_margin(right_aggregation_volume, csize, height, width)


    print('\nSelecting best disparities...')
    left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume), max_disparity))
    right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), max_disparity))


    print('\nApplying median filter...')
    left_disparity_map = cv2.medianBlur(left_disparity_map, bsize[0])
    right_disparity_map = cv2.medianBlur(right_disparity_map, bsize[0])

    print("\nFiltering disparities...")
    left_filtered_disp = filter_disparity(left_disparity_map, left_confidences, conf_threshold)
    right_filtered_disp = filter_disparity(right_disparity_map, right_confidences, conf_threshold)
    
    if save_images:
        cv2.imwrite(f'left_{output_name}', colorize_image(left_disparity_map, "jet"))
        cv2.imwrite(f'right_{output_name}', colorize_image(right_disparity_map, "jet"))
        cv2.imwrite(f'left_conf_{output_name}', colorize_image(np.uint8(left_confidences * 255), "winter"))
        cv2.imwrite(f'right_conf_{output_name}', colorize_image(np.uint8(right_confidences * 255), "winter"))
        cv2.imwrite(f'left_filtered_{output_name}', colorize_image(left_filtered_disp, "jet"))
        cv2.imwrite(f'right_filtered_{output_name}', colorize_image(right_filtered_disp, "jet"))


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