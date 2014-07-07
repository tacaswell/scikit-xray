# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
This module is for spectroscopy specific tools (spectrum fitting etc).
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import zip
import numpy as np
from .fitting.methods import fit_quad_to_peak


def align_and_scale(energy_list, counts_list, pk_find_fun=None):
    """

    Parameters
    ----------
    energy_list : iterable of ndarrays
        list of ndarrays with the energy of each element

    counts_list : iterable of ndarrays
        list of ndarrays of counts/element

    pk_find_fun : function or None
       A function which takes two ndarrays and returns parameters
       about the largest peak.  If None, defaults to `find_largest_peak`.
       For this demo, the output is (center, height, width), but this sould
       be pinned down better.

    Returns
    -------
    out_e : list of ndarray
       The aligned/scaled energy arrays

    out_c : list of ndarray
       The count arrays (should be the same as the input)
    """
    if pk_find_fun is None:
        pk_find_fun = find_larest_peak

    base_sigma = None
    out_e, out_c = [], []
    for e, c in zip(energy_list, counts_list):
        E0, max_val, sigma = pk_find_fun(e, c)
        print(E0, max_val, sigma)
        if base_sigma is None:
            base_sigma = sigma
        out_e.append((e - E0) * base_sigma / sigma)
        out_c.append(c)

    return out_e, out_c


def find_larest_peak(X, Y, window=5):
    """
    Finds and estimates the location, width, and height of
    the largest peak. Assumes the top of the peak can be
    approximated as a Gaussian.  Finds the peak properties
    using least-squares fitting of a parabola to the log of
    the counts.

    The region around the peak can be approximated by
    Y = Y0 * exp(- (X - X0)**2 / (2 * sigma **2))

    Parameters
    ----------
    X : ndarray
       The independent variable

    Y : ndarary
      Dependent variable sampled at positions X

    window : int, optional
       The size of the window around the maximum to use
       for the fitting


    Returns
    -------
    X0 : float
        The location of the peak

    Y0 : float
        The magnitude of the peak

    sigma : float
        Width of the peak
    """

    # make sure they are _really_ arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # get the bin with the largest number of counts
    j = np.argmax(Y)
    roi = slice(np.max(j - window, 0),
                j + window + 1)

    (w, X0, Y0), R2 = fit_quad_to_peak(X[roi],
                                        np.log(Y[roi]))

    return X0, np.exp(Y0), 1/np.sqrt(-2*w)
