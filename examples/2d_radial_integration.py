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
'''
Created on Jun 4, 2014
'''

import matplotlib.pyplot as plt
import numpy as np
from skxray.io.binary import read_binary
from skxray.core import pixel_to_radius


def get_cbr4_sample_img():
    # define the
    fname = "data/2d/cbr4_singlextal_rotate190_50deg_2s_90kev_203f.cor.042.cor"
    params = {"filename": fname,
            "nx": 2048,
            "ny": 2048,
            "nz": 1,
            "headersize": 0,
            "dsize": np.uint16,
            # these numbers come from
            # https://github.com/JamesDMartin/RamDog/blob/master/Calibration/APS--2009--CeO2.calib
            "wavelength": .13702,
            "detector_center": (1033.321, 1020.208),
            "dist_sample": 188.672,
            "pixel_size": (.200, .200)
            }

    # read in a binary file
    data, header = read_binary(**params)

    return data, params


def integrate_and_plot(data, params):
    """
    Integrate and plot a powder diffraction image
    """
    # convert the data from 2d array to xyi relative to beam center

    R = pixel_to_radius(data.shape,
                        params['detector_center'],
                        pixel_size=params['pixel_size'])

    I = data.ravel()

    # bin i based on r
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(R, I, lw=3, color='k')
    ax2.imshow(data, cmap='gray')
    cx, cy = params['detector_center']
    ax2.axvline(cx, color='r')
    ax2.axhline(cy, color='r')

    plt.show()


if __name__ == "__main__":
    # get the sample data
    data, params = get_cbr4_sample_img()
    integrate_and_plot(data, params)
