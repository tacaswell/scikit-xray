# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
"""
This module is for generic fitting methods
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import scipy.special
import scipy.signal

from scipy.special import erf, erfc


def gauss_peak(gain, sigma, delta_energy):
    """
    TODO: this needs numpy-style docstring

    models a gaussian fluorescence peak
    please refer to van espen, spectrum evaluation in van grieken,
    handbook of x-ray spectrometry, 2nd ed, page 182 ff
    """
    pre_factor = gain/(sigma * np.sqrt(2.*np.pi))
    counts = pre_factor * np.exp(-((delta_energy / (2*sigma))**2))

    return counts


def gauss_step(gain, sigma, delta_energy, peak_E):
    """
    TODO: this needs numpy-style docstring

    models a gaussian step fluorescence peak,
    please refer to van espen, spectrum evaluation,
    in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182 ff
    """

    # TODO this needs () to make it readable
    counts = gain / 2 / peak_E * erfc(delta_energy/(np.sqrt(2)*sigma))

    return counts


def gauss_tail(gain, sigma, delta_energy, gamma):
    """
    TODO: this needs numpy-style docstring

    models a gaussian fluorescence peak
    please see also van espen, spectrum evaluation,
    in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182 ff
    """

    delta_energy_neg = delta_energy.copy()
    wo_neg = (np.nonzero(delta_energy_neg > 0))[0]
    if wo_neg.size > 0:
        delta_energy_neg[wo_neg] = 0.
    temp_a = np.exp(delta_energy_neg / (gamma * sigma))
    # TODO this is utterly un-readable Split up into separate steps
    # and/or use lots of ()
    counts = (gain / 2. / gamma / sigma / np.exp(-0.5/(gamma**2)) *
        temp_a * erfc(delta_energy/(np.sqrt(2)*sigma)+(1./(gamma*np.sqrt(2)))))

    return counts


def elastic_peak(fitp, counts, ev, p, gain, matrix=False):
    """
    TODO: this needs numpy-style docstring

    elastic peak as a gaussian function
    """
    keywords = fitp.keywords
    sigma = np.sqrt((p[keywords.fwhm_pos[0]]/2.3548)**2 +
                    (p[keywords.coherent_pos[0]])*2.96*p[keywords.fwhm_pos[1]])
    delta_energy = ev - (p[keywords.coherent_pos[0]])

    # elastic peak, gaussian
    value = 1
    if not matrix:
        value = value * 10.**(p[keywords.coherent_pos[1]])
    value = value * gauss_peak(gain, sigma, delta_energy)
    counts = counts + value

    return counts, sigma


def compton_peak(fitp, counts, ev, p, gain, matrix=False):
    """
    TODO this needs numpy-style docstring
    """
    keywords = fitp.keywords
    # TODO this is unreadable, fix it
    # TODO remove the magic numbers
    compton_E = p[keywords.coherent_pos[0]]/(1+(
        p[keywords.coherent_pos[0]]/511.) *
        (1-np.cos(p[keywords.compton_pos[0]]*2*np.pi/360)))

    sigma = np.sqrt((p[keywords.fwhm_pos[0]]/2.3548)**2 +
                    compton_E*2.96*p[keywords.fwhm_pos[1]])

    # TODO this is never used, why is it here?
    local_sigma = sigma*p[14]

    delta_energy = ev.copy() - compton_E

    # compton peak, gaussian
    faktor = 1. / (1+p[keywords.compton_pos[3]] +
                   p[keywords.compton_pos[4]] +
                   p[keywords.compton_pos[6]])

    if not matrix:
        faktor = faktor * (10.**p[keywords.compton_pos[2]])

    value = faktor * gauss_peak(gain,
                                sigma*p[keywords.compton_pos[1]],
                                delta_energy)
    counts = counts + value

    # compton peak, step
    if p[keywords.compton_pos[3]] > 0.:
        value = faktor * p[keywords.compton_pos[3]]
        value = value * gauss_step(gain, sigma, delta_energy, compton_E)
        counts = counts + value

    # compton peak, tail on the low side
    value = faktor * p[keywords.compton_pos[4]]
    value = value * gauss_tail(gain, sigma,
                               delta_energy, p[keywords.compton_pos[5]])
    counts = counts + value

    # compton peak, tail on the high side
    value = faktor * p[keywords.compton_pos[6]]
    value = value * gauss_tail(gain, sigma,
                               -delta_energy, p[keywords.compton_pos[7]])
    counts = counts + value

    return counts, sigma, faktor


def maps_snip(fitp, par_values):
    """
    TODO this need documentation

    generate background
    """
    # TODO this is never used, why is it here?
    v = 1

    keywords = fitp.keywords

    background = keywords.spectrum.copy()
    n_background = background.size

    # calculate the energy axis from parameter values

    e_off = par_values[keywords.energy_pos[0]]
    e_lin = par_values[keywords.energy_pos[1]]
    e_quad = par_values[keywords.energy_pos[2]]

    energy = np.arange(np.float(n_background))
    if keywords.spectral_binning > 0:
        energy = energy * keywords.spectral_binning

    energy = e_off + energy * e_lin + np.power(energy, 2) * e_quad

    tmp = (e_off/2.3548)**2 + energy*2.96*e_lin
    wind = np.nonzero(tmp < 0)[0]
    tmp[wind] = 0.
    fwhm = 2.35 * np.sqrt(tmp)

    # TODO this is never used, why is it here?
    original_bcgrd = background.copy()

    # smooth the background
    if keywords.spectral_binning > 0:
        s = scipy.signal.boxcar(3)
    else:
        s = scipy.signal.boxcar(5)
    A = s.sum()
    background = scipy.signal.convolve(background, s, mode='same')/A

    # SNIP PARAMETERS
    window_rf = np.sqrt(2)

    width = par_values[keywords.added_params[0]]

    window_p = width * fwhm / e_lin  # in channels
    if keywords.spectral_binning > 0:
        window_p = window_p/2.

    background = np.log(np.log(background+1.)+1.)

    index = np.arange(np.float(n_background))
    # FIRST SNIPPING

    if keywords.spectral_binning > 0:
        no_iterations = 3
    else:
        no_iterations = 2

    for j in range(no_iterations):
        lo_index = index - window_p
        wo = np.where(lo_index < max((keywords.xmin, 0)))
        lo_index[wo] = max((keywords.xmin, 0))
        hi_index = index + window_p
        wo = np.where(hi_index > min((keywords.xmax, n_background-1)))
        hi_index[wo] = min((keywords.xmax, n_background-1))

        temp = (background[lo_index.astype(np.int)] +
                background[hi_index.astype(np.int)]) / 2

        wo = np.where(background > temp)
        background[wo] = temp[wo]

    if keywords.spectral_binning > 0:
        no_iterations = 7
    else:
        no_iterations = 12

    current_width = window_p
    max_current_width = np.amax(current_width)

    while max_current_width >= 0.5:
        lo_index = index - current_width
        wo = np.where(lo_index < max((keywords.xmin, 0)))
        lo_index[wo] = max((keywords.xmin, 0))
        hi_index = index + current_width
        wo = np.where(hi_index > min((keywords.xmax, n_background-1)))
        hi_index[wo] = min((keywords.xmax, n_background-1))

        temp = (background[lo_index.astype(np.int)] +
                background[hi_index.astype(np.int)]) / 2
        wo = np.where(background > temp)
        background[wo] = temp[wo]

        current_width = current_width / window_rf
        max_current_width = np.amax(current_width)

    background = np.exp(np.exp(background)-1.)-1.

    wo = np.where(not np.isfinite(background))
    background[wo] = 0.

    keywords.background = background.copy()

    return background
