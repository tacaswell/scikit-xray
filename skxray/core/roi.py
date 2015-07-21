#! encoding: utf-8
# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

"""
This module contain convenience methods to generate ROI labeled arrays for
simple shapes such as rectangles and concentric circles.
"""
from __future__ import absolute_import, division, print_function

import collections
import logging
import scipy.ndimage.measurements as ndim

import numpy as np

from . import utils

logger = logging.getLogger(__name__)


def rectangles(coords, shape):
    """
    This function wil provide the indices array for rectangle region of
    interests.

    Parameters
    ----------
    coords : iterable
        coordinates of the upper-left corner and width and height of each
        rectangle: e.g., [(x, y, w, h), (x, y, w, h)]

    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).

    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in coords. Order is (rr, cc).

    """

    labels_grid = np.zeros(shape, dtype=np.int64)

    for i, (col_coor, row_coor, col_val, row_val) in enumerate(coords):

        left, right = np.max([col_coor, 0]), np.min([col_coor + col_val,
                                                     shape[0]])
        top, bottom = np.max([row_coor, 0]), np.min([row_coor + row_val,
                                                     shape[1]])

        slc1 = slice(left, right)
        slc2 = slice(top, bottom)

        if np.any(labels_grid[slc1, slc2]):
            raise ValueError("overlapping ROIs")

        # assign a different scalar for each roi
        labels_grid[slc1, slc2] = (i + 1)

    return labels_grid


def rings(edges, center, shape):
    """
    Draw annual (ring-shaped) regions of interest.

    Each ring will be labeled with an integer. Regions outside any ring will
    be filled with zeros.

    Parameters
    ----------
    edges: list
        giving the inner and outer radius of each ring
        e.g., [(1, 2), (11, 12), (21, 22)]

    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).

    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).

    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in edges.
    """
    edges = np.atleast_2d(np.asarray(edges)).ravel()
    if not 0 == len(edges) % 2:
        raise ValueError("edges should have an even number of elements, "
                         "giving inner, outer radii for each ring")
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("edges are expected to be monotonically increasing, "
                         "giving inner and outer radii of each ring from "
                         "r=0 outward")
    r_coord = utils.radial_grid(center, shape).ravel()
    label_array = np.digitize(r_coord, edges, right=False)
    # Even elements of label_array are in the space between rings.
    label_array = (np.where(label_array % 2 != 0, label_array, 0) + 1) // 2
    return label_array.reshape(shape)


def ring_edges(inner_radius, width, spacing=0, num_rings=None):
    """
    Calculate the inner and outer radius of a set of rings.

    The number of rings, their widths, and any spacing between rings can be
    specified. They can be uniform or varied.

    Parameters
    ----------
    inner_radius : float
        inner radius of the inner-most ring

    width : float or list of floats
        ring thickness
        If a float, all rings will have the same thickness.

    spacing : float or list of floats, optional
        margin between rings, 0 by default
        If a float, all rings will have the same spacing. If a list,
        the length of the list must be one less than the number of
        rings.

    num_rings : int, optional
        number of rings
        Required if width and spacing are not lists and number
        cannot thereby be inferred. If it is given and can also be
        inferred, input is checked for consistency.

    Returns
    -------
    edges : array
        inner and outer radius for each ring

    Example
    -------
    # Make two rings starting at r=1px, each 5px wide
    >>> ring_edges(inner_radius=1, width=5, num_rings=2)
    [(1, 6), (6, 11)]
    # Make three rings of different widths and spacings.
    # Since the width and spacings are given individually, the number of
    # rings here is simply inferred.
    >>> ring_edges(inner_radius=1, width=(5, 4, 3), spacing=(1, 2))
    [(1, 6), (7, 11), (13, 16)]
    """
    # All of this input validation merely checks that width, spacing, and
    # num_rings are self-consistent and complete.
    width_is_list = isinstance(width, collections.Iterable)
    spacing_is_list = isinstance(spacing, collections.Iterable)
    if (width_is_list and spacing_is_list):
        if len(width) != len(spacing) - 1:
            raise ValueError("List of spacings must be one less than list "
                             "of widths.")
    if num_rings is None:
        try:
            num_rings = len(width)
        except TypeError:
            try:
                num_rings = len(spacing) + 1
            except TypeError:
                raise ValueError("Since width and spacing are constant, "
                                 "num_rings cannot be inferred and must be "
                                 "specified.")
    else:
        if width_is_list:
            if num_rings != len(width):
                raise ValueError("num_rings does not match width list")
        if spacing_is_list:
            if num_rings-1 != len(spacing):
                raise ValueError("num_rings does not match spacing list")

    # Now regularlize the input.
    if not width_is_list:
        width = np.ones(num_rings) * width
    if not spacing_is_list:
        spacing = np.ones(num_rings - 1) * spacing

    # The inner radius is the first "spacing."
    all_spacings = np.insert(spacing, 0, inner_radius)
    steps = np.array([all_spacings, width]).T.ravel()
    edges = np.cumsum(steps).reshape(-1, 2)

    return edges


def segmented_rings(edges, segments, center, shape, offset_angle=0):
    """
    Parameters
    ----------
    edges : array
         inner and outer radius for each ring

    segments : int or list
        number of pie slices or list of angles in radians
        That is, 8 produces eight equal-sized angular segments,
        whereas a list can be used to produce segments of unequal size.

    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).

    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).

    angle_offset : float or array, optional
        offset in radians from offset_angle=0 along the positive X axis

    Returns
    -------
    label_array : array
        Elements not inside any ROI are zero; elements inside each
        ROI are 1, 2, 3, corresponding to the order they are specified
        in edges and segments

    """
    edges = np.asarray(edges).ravel()
    if not 0 == len(edges) % 2:
        raise ValueError("edges should have an even number of elements, "
                         "giving inner, outer radii for each ring")
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("edges are expected to be monotonically increasing, "
                         "giving inner and outer radii of each ring from "
                         "r=0 outward")

    agrid = utils.angle_grid(center, shape)

    agrid[agrid < 0] = 2*np.pi + agrid[agrid < 0]

    segments_is_list = isinstance(segments, collections.Iterable)
    if segments_is_list:
        segments = np.asarray(segments) + offset_angle
    else:
        # N equal segments requires N+1 bin edges spanning 0 to 2pi.
        segments = np.linspace(0, 2*np.pi, num=1+segments, endpoint=True)
        segments += offset_angle

    # the indices of the bins(angles) to which each value in input
    #  array(angle_grid) belongs.
    ind_grid = (np.digitize(np.ravel(agrid), segments,
                            right=False)).reshape(shape)

    label_array = np.zeros(shape, dtype=np.int64)
    # radius grid for the image_shape
    rgrid = utils.radial_grid(center, shape)

    # assign indices value according to angles then rings
    len_segments = len(segments)
    for i in range(len(edges) // 2):
        indices = (edges[2*i] <= rgrid) & (rgrid < edges[2*i + 1])
        # Combine "segment #" and "ring #" to get unique label for each.
        label_array[indices] = ind_grid[indices] + (len_segments - 1) * i

    return label_array


def roi_max_counts(images_sets, label_array):
    """
    Return the brightest pixel in any ROI in any image in the image set.

    Parameters
    ----------
    images_sets : array
        iterable of 4D arrays
        shapes is: (len(images_sets), )

    label_array : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    max_counts : int
        maximum pixel counts
    """
    max_cts = 0
    for img_set in images_sets:
        for img in img_set:
            max_cts = max(max_cts, ndim.maximum(img, label_array))
    return max_cts


def roi_pixel_values(image, labels, index=None):
    """
    This will provide intensities of the ROI's of the labeled array
    according to the pixel list
    eg: intensities of the rings of the labeled array

    Parameters
    ----------
    image : array
        image data dimensions are: (rr, cc)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    index_list : list, optional
        labels list
        eg: 5 ROI's
        index = [1, 2, 3, 4, 5]

    Returns
    -------
    roi_pix : list
        intensities of the ROI's of the labeled array according
        to the pixel list

    """
    if labels.shape != image.shape:
        raise ValueError("Shape of the image data should be equal to"
                         " shape of the labeled array")
    if index is None:
        index = np.arange(1, np.max(labels) + 1)

    roi_pix = []
    for n in index:
        roi_pix.append(image[labels == n])
    return roi_pix, index


def mean_intensity_sets(images_set, labels):
    """
    Mean intensities for ROIS' of the labeled array for different image sets

    Parameters
    ----------
    images_set : array
        images sets
        shapes is: (len(images_sets), )
        one images_set is iterable of 2D arrays dimensions are: (rr, cc)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    Returns
    -------
    mean_intensity_list : list
        average intensity of each ROI as a list
        shape len(images_sets)

    index_list : list
        labels list for each image set

    """
    return tuple(map(list,
                     zip(*[mean_intensity(im,
                                          labels) for im in images_set])))


def mean_intensity(images, labels, index=None):
    """
    Mean intensities for ROIS' of the labeled array for set of images

    Parameters
    ----------
    images : array
        Intensity array of the images
        dimensions are: (num_img, num_rows, num_cols)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    index : list
        labels list
        eg: 5 ROI's
        index = [1, 2, 3, 4, 5]

    Returns
    -------
    mean_intensity : array
        mean intensity of each ROI for the set of images as an array
        shape (len(images), number of labels)

    """
    if labels.shape != images[0].shape[0:]:
        raise ValueError("Shape of the images should be equal to"
                         " shape of the label array")
    if index is None:
        index = np.arange(1, np.max(labels) + 1)

    mean_intensity = np.zeros((images.shape[0], index.shape[0]))
    for n, img in enumerate(images):
        mean_intensity[n] = ndim.mean(img, labels, index=index)

    return mean_intensity, index


def combine_mean_intensity(mean_int_list, index_list):
    """
    Combine mean intensities of the images(all images sets) for each ROI
    if the labels list of all the images are same

    Parameters
    ----------
    mean_int_list : list
        mean intensity of each ROI as a list
        shapes is: (len(images_sets), )

    index_list : list
        labels list for each image sets

    img_set_names : list

    Returns
    -------
    combine_mean_int : array
        combine mean intensities of image sets for each ROI of labeled array
        shape (number of images in all image sets, number of labels)

    """
    if np.all(map(lambda x: x == index_list[0], index_list)):
        combine_mean_intensity = np.vstack(mean_int_list)
    else:
        raise ValueError("Labels list for the image sets are different")

    return combine_mean_intensity


def circular_average(image, calibrated_center, threshold=0, nx=100,
                     pixel_size=None):
    """
    Circular average(radial integration) of the intensity distribution of
    the image data.

    Parameters
    ----------
    image : array
        input image

    calibrated_center : tuple
        The center in pixels-units (row, col)

    threshold : int, optional
        threshold value to mask

    nx : int, optional
        number of bins

    pixel_size : tuple, optional
        The size of a pixel in real units. (height, width). (mm)

    Returns
    -------
    bin_centers : array
        bin centers from bin edges
        shape [nx]

    ring_averages : array
        circular integration of intensity
    """
    radial_val = utils.radial_grid(calibrated_center, image.shape,
                                   pixel_size)

    bin_edges, sums, counts = utils.bin_1D(np.ravel(radial_val),
                                           np.ravel(image), nx)
    th_mask = counts > threshold
    ring_averages = sums[th_mask] / counts[th_mask]

    bin_centers = utils.bin_edges_to_centers(bin_edges)[th_mask]

    return bin_centers, ring_averages


def roi_kymograph(images, labels, num):
    """
    This function will provide data for graphical representation of pixels
    variation over time for required ROI.

    Parameters
    ----------
    images : array
        Intensity array of the images
        dimensions are: (num_img, num_rows, num_cols)

    labels : array
        labeled array; 0 is background.
        Each ROI is represented by a distinct label (i.e., integer).

    num : int
        required ROI label

    Returns
    -------
    roi_kymograph : array
        data for graphical representation of pixels variation over time
        for required ROI

    """
    roi_kymo = []
    for n, img in enumerate(images):
        roi_kymo.append((roi_pixel_values(img,
                                          labels == num)[0])[0])

    return np.matrix(roi_kymo)
