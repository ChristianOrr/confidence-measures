"""
All the functions required for evaluating the confidence measures.
"""
import numpy as np


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
    auc = np.trapezoid(roc, dx=1./20)
    optimal_auc = np.trapezoid(optimal_roc, dx=1./20)
    return auc, optimal_auc