import numpy as np
import sys
from functools import lru_cache

sys.path.append('../')
sys.path.append('./')
from utils.base_util import readNiiImage, saveNiiImage


def histogramMatching(pt_data, t_quantiles, t_values):
    # Stores the image data shape that will be used later
    oldshape = pt_data.shape
    # Converts the data arrays to single dimension and normalizes by the maximum
    pt_data_array = pt_data.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(pt_data_array,
                                            return_inverse=True,
                                            return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # Reshapes the corresponding values to the indexes and reshapes the array to input
    out_img = interp_t_values[bin_idx].reshape(oldshape)
    return out_img


# @lru_cache(200)
def matching(template, pt_data):

    nt_data_array = template.ravel()
    t_values, t_counts = np.unique(nt_data_array, return_counts=True)
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    out_img = histogramMatching(pt_data, t_quantiles, t_values)
    return out_img


def histForTrain(t_quantiles, t_values, pt_data):

    oldshape = pt_data.shape
    # Converts the data arrays to single dimension and normalizes by the maximum
    pt_data_array = pt_data.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(pt_data_array,
                                            return_inverse=True,
                                            return_counts=True)

    zero_index = np.where(s_values == 0)[0]
    # for i in range(len(bin_idx)):
    #     if bin_idx[i] == zero_index:
    #         bin_idx[i] = i

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    # Reshapes the corresponding values to the indexes and reshapes the array to input
    out_img = interp_t_values[bin_idx].reshape(oldshape)
    return out_img


def histForSave(t_quantiles, t_values, pt_data, dst, infos):
    out_img = histForTrain(t_quantiles, t_values, pt_data)
    saveNiiImage(out_img, infos, dst)
    print(f"--- {dst} done! ---")