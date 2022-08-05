"""
This is the python script version of the  
confidence estimation technique.
Very little explanations are provided for the methods in this script.
To see a detailed overview of this implementation see the jupyter
notebook with the same name.
"""

import argparse
import sys
import time as t
import cv2
import numpy as np
from matplotlib import cm


def get_path_cost(slice, offset, penalties, other_dim, disparity_dim):
    """
    Calculates the minimum costs for all potential disparities of 
    the pixels along a single path direction.
    Arguments:
        - slice: Array containing costs for all disparities, D, 
            along a direction, M, with dimension M x D
        - offset: Number of pixels on the border to ignore.
        - penalties: Matrix containing the penalties to assign to the 
            previous disparities costs. For previous disparities that differ 
            from current disparities. 
        - other_dim: Number of pixels in the current paths direction.
        - disparity_dim: Number of disparities to calculate minimum costs.

    Returns: The pixels minimum costs for all disparities, D, 
        along path direction, M, with shape M x D.
    """
    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=np.uint32)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for pixel_index in range(offset, other_dim):
        # Get all the minimum disparities costs from the previous pixel in the path
        previous_cost = minimum_cost_path[pixel_index - 1, :]
        # Get all the disparities costs (from the cost volume) for the current pixel
        current_cost = slice[pixel_index, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        # Add penalties to the previous pixels disparities that differ from current pixels disparities
        costs = costs + penalties
        # Find minimum costs for the current pixels disparities using the previous disparities costs + penalties 
        costs = np.amin(costs, axis=0)  
        # Current pixels disparities costs + minimum previous pixel disparities costs (with penalty) - 
        # (constant term) minimum previous cost from all disparities 
        pixel_direction_costs = current_cost + costs - np.amin(previous_cost)
        minimum_cost_path[pixel_index, :] = pixel_direction_costs

    return minimum_cost_path   


def get_penalties(max_disparity, P2, P1):
    """
    Creates a matrix of all the potential penalties for matching
    a current disparity (represented by the column index), with 
    a previous disparity (represented by the row index).
    Arguments:
        - max_disparity: Maximum disparity of the array.
        - P2: Penalty for disparity difference > 1
        - P1: Penalty for disparity difference = 1
    
    Return: Matrix containing all the penalties when disparity d1 from a column
            is matched with a previous disparity d2 from the row.
    """
    p2 = np.full(shape=(max_disparity, max_disparity), fill_value=P2, dtype=np.uint32)
    p1 = np.full(shape=(max_disparity, max_disparity), fill_value=P1 - P2, dtype=np.uint32)
    p1 = np.tril(p1, k=1) # keep values lower than k'th diagonal
    p1 = np.triu(p1, k=-1) # keep values higher than k'th diagonal
    no_penalty = np.identity(max_disparity, dtype=np.uint32) * -P1 # create diagonal matrix with values -p1
    penalties = p1 + p2 + no_penalty
    return penalties


def aggregate_costs(cost_volume, P2, P1, height, width, disparities):
    """
    Calculates the pixels costs for all disparities along all paths (4 in this case).

    Arguments: 
        - cost_volume: Array containing the matching cost for each pixel at each disparity.
        - P2: Penalty for disparity difference > 1
        - P1: Penalty for disparity difference = 1
        - height: Number of rows of the image.
        - width: Number of columns of the image.
        - disparities: Number of disparities to calculate minimum matching costs.

    Returns: Array containing the pixels matching costs for all disparities along 
        all directions, with dimension H x W x D X 4.
    """
    sys.stdout.flush()
    dawn = t.time()

    penalties = get_penalties(disparities, P2, P1)

    print("\tProcessing North and South aggregation...")
    south_aggregation = np.zeros(shape=(height, width, disparities), dtype=np.uint32)
    north_aggregation = np.copy(south_aggregation)

    for x in range(0, width):
        # Takes all the rows and disparities for a single column
        south = cost_volume[:, x, :]
        # Invert the rows to get the opposite direction
        north = np.flip(south, axis=0)
        south_aggregation[:, x, :] = get_path_cost(south, 1, penalties, height, disparities)
        north_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, penalties, height, disparities), axis=0)


    print("\tProcessing East and West aggregation...", end='')
    east_aggregation = np.copy(south_aggregation)
    west_aggregation = np.copy(south_aggregation)
    for y in range(0, height):
        # Takes all the column and disparities for a single row
        east = cost_volume[y, :, :]
        # Invert the columns to get the opposite direction
        west = np.flip(east, axis=0)
        east_aggregation[y, :, :] = get_path_cost(east, 1, penalties, width, disparities)
        west_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, penalties, width, disparities), axis=0)

    # Combine the costs from all paths into a single aggregation volume
    aggregation_volume = np.concatenate((south_aggregation[..., None], north_aggregation[..., None], east_aggregation[..., None], west_aggregation[..., None]), axis=3)

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return aggregation_volume


def compute_census(left, right, csize, height, width):
    """
    Calculate census bit strings for each pixel in the left and right images.
    Arguments:
        - left: left grayscale image.
        - right: right grayscale image.
        - csize: kernel size for the census transform.
        - height: number of rows of the image.
        - width: number of columns of the image.

    Return: Left and right images with pixel intensities replaced with census bit strings.
    """
    cheight = csize[0]
    cwidth = csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)

    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # offset is used since pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            # left
            center_pixel = left[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int32)
            image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            # If value is less than center value assign 1 otherwise assign 0 
            left_census_pixel_array = np.where(comparison < 0, 1, 0).flatten()
            # Convert census array to an integer by using bit shift operator
            left_census_pixel = np.int32(left_census_pixel_array.dot(1 << np.arange(cheight * cwidth)[::-1])) 
            left_census_values[y, x] = left_census_pixel

            # right
            center_pixel = right[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int32)
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            # If value is less than center value assign 1 otherwise assign 0 
            right_census_pixel_array = np.where(comparison < 0, 1, 0).flatten()
            # Convert census array to an integer by using bit shift operator
            right_census_pixel = np.int32(right_census_pixel_array.dot(1 << np.arange(cheight * cwidth)[::-1])) 
            right_census_values[y, x] = right_census_pixel

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_census_values, right_census_values


def compute_costs(left_census_values, right_census_values, max_disparity, csize, height, width):
    """
    Create cost volume for all potential disparities. 
    Cost volumes for both left and right images are calculated.
    Hamming distance is used to calculate the matching cost between 
    two pixels census values.
    Arguments:
        - left_census_values: left image containing census bit strings for each pixel (in integer form).
        - right_census_values: right image containing census bit strings for each pixel (in integer form).
        - max_disparity: maximum disparity to measure.
        - csize: kernel size for the census transform.
        - height: number of rows of the image.
        - width: number of columns of the image.

    Return: Left and right cost volumes with dimensions H x W x D.
    """
    cwidth = csize[1]
    x_offset = int(cwidth / 2)

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, max_disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, max_disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int32)
    rcensus = np.zeros(shape=(height, width), dtype=np.int32)

    for d in range(0, max_disparity):
        # The right image is shifted d pixels accross
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        # 1 is assigned when the bits differ and 0 when they are the same
        left_xor = np.int32(np.bitwise_xor(np.int32(left_census_values), rcensus))
        # All the 1's are summed up to give us the number of different pixels (the cost)
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        # All the costs for that disparity are added to the cost volume
        left_cost_volume[:, :, d] = left_distance

        # The left image is shifted d pixels accross
        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        # 1 is assigned when the bits differ and 0 when they are the same
        right_xor = np.int32(np.bitwise_xor(np.int32(right_census_values), lcensus))
        # All the 1's are summed up to give us the number of different pixels (the cost)
        right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
            right_distance[mask] = right_distance[mask] + 1
        # All the costs for that disparity are added to the cost volume
        right_cost_volume[:, :, d] = right_distance

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume


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


def sgm(left, right, max_disparity, P1, P2, csize, bsize):
    """
    Uses SGM to extract the aggregation cost volumes from 
    the left and right images.

    Arguments:
        - left: left image numpy array with dimensions H x W.
        - right: right image numpy array with dimensions H x W.
        - max_disparity: maximum disparity of the array.
        - P2: Penalty for disparity difference > 1.
        - P1: Penalty for disparity difference = 1.
        - csize: kernel size for the census transform.
        - bsize: kernel size for gaussian blur.

    Returns: Left and right aggregation cost volumes
    """
    print("Performing Gaussian blur on the images...")
    left = cv2.GaussianBlur(left, bsize, 0, 0)
    right = cv2.GaussianBlur(right, bsize, 0, 0)

    print('\nStarting cost computation...')
    left_census, right_census = compute_census(left, right, csize, height, width)
    left_cost_volume, right_cost_volume = compute_costs(left_census, right_census, max_disparity, csize, height, width)


    print('\nStarting left aggregation computation...')
    left_aggregation_volume = aggregate_costs(left_cost_volume, P2, P1, height, width, max_disparity)
    print('\nStarting right aggregation computation...')
    right_aggregation_volume = aggregate_costs(right_cost_volume, P2, P1, height, width, max_disparity)


    return left_aggregation_volume, right_aggregation_volume



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='cones/im2.png', help='name (path) to the left image')
    parser.add_argument('--right', default='cones/im6.png', help='name (path) to the right image')
    parser.add_argument('--left_gt', default='cones/disp2.png', help='name (path) to the left ground-truth image')
    parser.add_argument('--right_gt', default='cones/disp6.png', help='name (path) to the right ground-truth image')
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


    left_aggregation_volume, right_aggregation_volume = sgm(left, right, max_disparity, P1, P2, csize, bsize)


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