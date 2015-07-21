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
from __future__ import absolute_import, division, print_function
import logging

import numpy as np

logger = logging.getLogger(__name__)

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)

from nose.tools import assert_equal, assert_true, assert_raises

import skxray.core.roi as roi
import skxray.core.correlation as corr
import skxray.core.utils as core

from skimage import morphology


def test_rectangles():
    shape = (15, 26)
    roi_data = np.array(([2, 2, 6, 3], [6, 7, 8, 5], [8, 18, 5, 10]),
                        dtype=np.int64)

    all_roi_inds = roi.rectangles(roi_data, shape)

    roi_inds, pixel_list = corr.extract_label_indices(all_roi_inds)

    ty = np.zeros(shape).ravel()
    ty[pixel_list] = roi_inds
    num_pixels_m = (np.bincount(ty.astype(int)))[1:]

    re_mesh = ty.reshape(*shape)
    for i, (col_coor, row_coor, col_val, row_val) in enumerate(roi_data, 0):
        ind_co = np.column_stack(np.where(re_mesh == i + 1))

        left, right = np.max([col_coor, 0]), np.min([col_coor + col_val,
                                                     shape[0]])
        top, bottom = np.max([row_coor, 0]), np.min([row_coor + row_val,
                                                     shape[1]])
        assert_almost_equal(left, ind_co[0][0])
        assert_almost_equal(right-1, ind_co[-1][0])
        assert_almost_equal(top, ind_co[0][1])
        assert_almost_equal(bottom-1, ind_co[-1][-1])


def test_rings():
    center = (100., 100.)
    img_dim = (200, 205)
    first_q = 10.
    delta_q = 5.
    num_rings = 7  # number of Q rings
    one_step_q = 5.0
    step_q = [2.5, 3.0, 5.8]

    # test when there is same spacing between rings
    edges = roi.ring_edges(first_q, width=delta_q, spacing=one_step_q,
                           num_rings=num_rings)
    print("edges there is same spacing between rings ", edges)
    label_array = roi.rings(edges, center, img_dim)
    print("label_array there is same spacing between rings", label_array)
    label_mask, pixel_list = corr.extract_label_indices(label_array)
    # number of pixels per ROI
    num_pixels = np.bincount(label_mask, minlength=(np.max(label_mask)+1))
    num_pixels = num_pixels[1:]

    # test when there is same spacing between rings
    edges = roi.ring_edges(first_q, width=delta_q, spacing=2.5,
                           num_rings=num_rings)
    print("edges there is same spacing between rings ", edges)
    label_array = roi.rings(edges, center, img_dim)
    print("label_array there is same spacing between rings", label_array)
    label_mask, pixel_list = corr.extract_label_indices(label_array)
    # number of pixels per ROI
    num_pixels = np.bincount(label_mask, minlength=(np.max(label_mask)+1))
    num_pixels = num_pixels[1:]

    # test when there is different spacing between rings
    edges = roi.ring_edges(first_q, width=delta_q, spacing=step_q,
                           num_rings=4)
    print("edges when there is different spacing between rings", edges)
    label_array = roi.rings(edges, center, img_dim)
    print("label_array there is different spacing between rings", label_array)
    label_mask, pixel_list = corr.extract_label_indices(label_array)
    # number of pixels per ROI
    num_pixels = np.bincount(label_mask, minlength=(np.max(label_mask)+1))
    num_pixels = num_pixels[1:]

    # test when there is no spacing between rings
    edges = roi.ring_edges(first_q, width=delta_q, num_rings=num_rings)
    print("edges", edges)
    label_array = roi.rings(edges, center, img_dim)
    print("label_array", label_array)
    label_mask, pixel_list = corr.extract_label_indices(label_array)
    # number of pixels per ROI
    num_pixels = np.bincount(label_mask, minlength=(np.max(label_mask)+1))
    num_pixels = num_pixels[1:]

    # Did we draw the right number of rings?
    print(np.unique(label_array))
    actual_num_rings = len(np.unique(label_array)) - 1
    assert_equal(actual_num_rings, num_rings)

    # Does each ring have more pixels than the last, being larger?
    ring_areas = np.bincount(label_array.ravel())[1:]
    area_comparison = np.diff(ring_areas)
    print(area_comparison)
    areas_monotonically_increasing = np.all(area_comparison > 0)
    assert_true(areas_monotonically_increasing)

    # Test various illegal inputs
    assert_raises(ValueError,
                  lambda: roi.ring_edges(1, 2))  # need num_rings
    # width incompatible with num_rings
    assert_raises(ValueError,
                  lambda: roi.ring_edges(1, [1, 2, 3], num_rings=2))
    # too few spacings
    assert_raises(ValueError,
                  lambda: roi.ring_edges(1, [1, 2, 3], [1]))
    # too many spacings
    assert_raises(ValueError,
                  lambda: roi.ring_edges(1, [1, 2, 3], [1, 2, 3]))
    # num_rings conflicts with width, spacing
    assert_raises(ValueError,
                  lambda: roi.ring_edges(1, [1, 2, 3], [1, 2], 5))


def _helper_check(pixel_list, inds, num_pix, edges, center,
                  img_dim, num_qs):
    # recreate the indices using pixel_list and inds values
    ty = np.zeros(img_dim).ravel()
    ty[pixel_list] = inds
    data = ty.reshape(img_dim[0], img_dim[1])

    # get the grid values from the center
    grid_values = core.radial_grid(img_dim, center)

    # get the indices into a grid
    zero_grid = np.zeros((img_dim[0], img_dim[1]))
    for r in range(num_qs):
        vl = (edges[r][0] <= grid_values) & (grid_values < edges[r][1])
        zero_grid[vl] = r + 1

    # check the num_pixels
    num_pixels = []
    for r in range(num_qs):
        num_pixels.append(int((np.histogramdd(np.ravel(grid_values), bins=1,
                                              range=[[edges[r][0],
                                                      (edges[r][1]
                                                       - 0.000001)]]))[0][0]))
    assert_array_equal(num_pix, num_pixels)


def test_segmented_rings():
    center = (75, 75)
    img_dim = (150, 140)
    first_q = 5
    delta_q = 5
    num_rings = 4  # number of Q rings
    slicing = 4

    edges = roi.ring_edges(first_q, width=delta_q, spacing=4,
                           num_rings=num_rings)
    print("edges", edges)

    label_array = roi.segmented_rings(edges, slicing, center,
                                      img_dim, offset_angle=0)
    print("label_array for segmented_rings", label_array)

    # Did we draw the right number of ROIs?
    label_list = np.unique(label_array.ravel())
    actual_num_labels = len(label_list) - 1
    num_labels = num_rings * slicing
    assert_equal(actual_num_labels, num_labels)

    # Did we draw the right ROIs? (1-16 with some zeros around too)
    assert_array_equal(label_list, np.arange(num_labels + 1))

    # A brittle test to make sure the exactly number of pixels per label
    # is never accidentally changed:
    # number of pixels per ROI
    num_pixels = np.bincount(label_array.ravel())
    expected_num_pixels = [18372, 59, 59, 59, 59, 129, 129, 129,
                           129, 200, 200, 200, 200, 269, 269, 269, 269]
    assert_array_equal(num_pixels, expected_num_pixels)


def test_roi_pixel_values():
    images = morphology.diamond(8)
    # width incompatible with num_rings

    label_array = np.zeros((256, 256))

    # different shapes for the images and labels
    assert_raises(ValueError,
                  lambda: roi.roi_pixel_values(images, label_array))
    # create a label mask
    center = (8., 8.)
    inner_radius = 2.
    width = 1
    spacing = 1
    edges = roi.ring_edges(inner_radius, width, spacing, num_rings=5)
    rings = roi.rings(edges, center, images.shape)

    intensity_data, index = roi.roi_pixel_values(images, rings)
    assert_array_equal(intensity_data[0], ([1, 1, 1, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1, 1, 1]))
    assert_array_equal([1, 2, 3, 4, 5], index)


def test_roi_max_counts():
    img_stack1 = np.random.randint(0, 60, size=(50, ) + (50, 50))
    img_stack2 = np.random.randint(0, 60, size=(100, ) + (50, 50))

    img_stack1[0][20, 20] = 60

    samples = (img_stack1, img_stack2)

    label_array = np.zeros((img_stack1[0].shape))

    label_array[img_stack1[0] < 20] = 1
    label_array[img_stack1[0] > 40] = 2

    assert_array_equal(60, roi.roi_max_counts(samples, label_array))


def test_static_test_sets():
    img_stack1 = np.random.randint(0, 60, size=(50, ) + (50, 50))

    label_array = np.zeros((25, 25))

    # different shapes for the images and labels
    assert_raises(ValueError,
                  lambda: roi.mean_intensity(img_stack1, label_array))
    images1 = []
    for i in range(10):
        int_array = np.tril(i*np.ones(50))
        int_array[int_array == 0] = i*100
        images1.append(int_array)

    images2 = []
    for i in range(20):
        int_array = np.triu(i*np.ones(50))
        int_array[int_array == 0] = i*100
        images2.append(int_array)

    samples = np.array((np.asarray(images1), np.asarray(images2)))

    roi_data = np.array(([2, 30, 12, 15], [40, 20, 15, 10]), dtype=np.int64)

    label_array = roi.rectangles(roi_data, shape=(50, 50))

    # test mean_intensity function
    average_intensity, index = roi.mean_intensity(np.asarray(images1),
                                                  label_array)
    # test mean_intensity_sets function
    average_int_sets, index_list = roi.mean_intensity_sets(samples,
                                                           label_array)

    assert_array_equal((list(average_int_sets)[0][:, 0]),
                       [float(x) for x in range(0, 1000, 100)])
    assert_array_equal((list(average_int_sets)[1][:, 0]),
                       [float(x) for x in range(0, 20, 1)])

    assert_array_equal((list(average_int_sets)[0][:, 1]),
                       [float(x) for x in range(0, 10, 1)])
    assert_array_equal((list(average_int_sets)[1][:, 1]),
                       [float(x) for x in range(0, 2000, 100)])

    # test combine_mean_intensity function
    combine_mean_int = roi.combine_mean_intensity(average_int_sets,
                                                  index_list)

    roi_data2 = np.array(([2, 30, 12, 15], [40, 20, 15, 10],
                          [20, 2, 4, 5]), dtype=np.int64)

    label_array2 = roi.rectangles(roi_data2, shape=(50, 50))

    average_int2, index2 = roi.mean_intensity(np.asarray(images1),
                                              label_array2)
    index_list2 = [index_list, index2]

    average_int_sets.append(average_int2)

    # raise ValueError when there is different labels in different image sets
    #  when trying to combine the values
    assert_raises(ValueError,
                  lambda: roi.combine_mean_intensity(average_int_sets,
                                                     index_list2))


def test_circular_average():
    image = np.zeros((12, 12))
    calib_center = (5, 5)
    inner_radius = 1

    edges = roi.ring_edges(inner_radius, width=1, spacing=1, num_rings=2)
    labels = roi.rings(edges, calib_center, image.shape)
    image[labels == 1] = 10
    image[labels == 2] = 10
    bin_cen, ring_avg = roi.circular_average(image, calib_center, nx=6)

    assert_array_almost_equal(bin_cen, [0.70710678, 2.12132034,
                                        3.53553391,  4.94974747,  6.36396103,
                                        7.77817459], decimal=6)
    assert_array_almost_equal(ring_avg, [8., 2.5, 5.55555556, 0.,
                                         0., 0.], decimal=6)


def test_roi_kymograph():
    calib_center = (25, 25)
    inner_radius = 5

    edges = roi.ring_edges(inner_radius, width=2, num_rings=1)
    labels = roi.rings(edges, calib_center, (50, 50))

    images = []
    for i in range(100):
        int_array = i*np.ones(labels.shape)
        images.append(int_array)

    kymograph_data = roi.roi_kymograph(np.asarray(images), labels, num=1)

    assert_almost_equal(kymograph_data[:, 0],  np.arange(100).reshape(100, 1))
