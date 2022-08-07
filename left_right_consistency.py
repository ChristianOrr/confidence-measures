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
from sgm_numpy import sgm
from evaluation_funcs import calc_roc, calc_auc
from helper_funcs import filter_disparity, select_disparity, normalize, colorize_image



def left_right_consistency(left_disp, right_disp, csize, height, width, left_is_reference=True):
    """
    Creates a confidence map (H x W) using the left-right 
    consistency technique from the left and right disparity maps (H x W).
    The confidence map is calculated for the left image by setting left_is_reference = True and
    for the right image by setting left_is_reference = False.

    Arguments:
        - left_disp: Array containing the left image disparities, with dimension H x W.
        - right_disp: Array containing the right image disparities, with dimension H x W.
        - csize: kernel size for the census transform.
        - height: number of rows of the image.
        - width: number of columns of the image.
        - left_is_reference: Bool, specifying whether left or right image is the reference image.
            The confidence map will only be calculated for the reference image.
        
    Returns: Confidence map with normalized values (0-1) and 
        dimensions H x W.
    """
    # Create an array containing the x-coords for each pixel
    x_coords = np.repeat(np.arange(width)[None, :], repeats=height, axis=0)
    # Calculate left image confidences
    if left_is_reference:
        new_x_coords = x_coords - left_disp
        # Prevent x indices going past the left border
        new_x_coords[new_x_coords < 0] = 0
        # Gets the disparities in the right image for pixels with x coords = x - d1
        new_right_disparities = np.take_along_axis(right_disp, indices=new_x_coords, axis=1)
        # Calculate LRC
        lrc = -np.abs(left_disp - new_right_disparities)
    # Calculate right image confidences
    else:
        new_x_coords = x_coords + right_disp
        # Prevent x indices going past the right border
        new_x_coords[new_x_coords >= width] = width - 1
        # Gets the disparities in the left image for pixels with x coords = x + d1
        new_left_disparities = np.take_along_axis(left_disp, indices=new_x_coords, axis=1)
        # Calculate LRC
        lrc = -np.abs(right_disp - new_left_disparities)
    # Normalize to get confidence values between 0-1
    minimum = np.min(lrc)
    maximum = np.max(lrc)
    confidence = (lrc - minimum) / (maximum - minimum)
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
    parser.add_argument('--conf_threshold', default=0.95, type=float, help='Retain disparitys with confidence greater than the conf_threshold')
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

    left_disparity = select_disparity(left_aggregation_volume)
    right_disparity = select_disparity(right_aggregation_volume)

    print("\nCalculating confidences...")
    left_confidences = left_right_consistency(left_disparity, right_disparity, csize, height, width, left_is_reference=True)
    right_confidences = left_right_consistency(left_disparity, right_disparity, csize, height, width, left_is_reference=False)


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