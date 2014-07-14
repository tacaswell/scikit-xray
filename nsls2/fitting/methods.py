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
from six.moves import zip
import numpy as np
from scipy.optimize import leastsq

# =============================================================================
# The code is this block is derived from code with the following license
# (BSD to Jonathan J. Helmus, no year and no organization)
#
# Copyright (c) Jonathan J. Helmus (jjhelmus@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#     Neither the name of the <ORGANIZATION> nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

# TODO re-factor this code use use closures instead of a
# zillion helper functions


def leastsqbound(func, x0, bounds, args=None, **kw):
    """
    Constrained multivariant Levenberg-Marquard optimization
    <OWNER> =

    Minimize the sum of squares of a given function using the
    Levenberg-Marquard algorithm. Contraints on parameters are inforced using
    variable transformations as described in the MINUIT User's Guide by
    Fred James and Matthias Winkler.

    Parameters:

    * func      functions to call for optimization.
    * x0        Starting estimate for the minimization.
    * bounds    (min,max) pair for each element of x, defining the bounds on
                that parameter.  Use None for one of min or max when there is
                no bound in that direction.
    * args      Any extra arguments to func are places in this tuple.

    Returns: (x,{cov_x,infodict,mesg},ier)

    Return is described in the scipy.optimize.leastsq function.  x and con_v
    are corrected to take into account the parameter transformation, infodic
    is not corrected.

    Additional keyword arguments are passed directly to the
    scipy.optimize.leastsq algorithm.

    """
    # check for full output
    full = ("full_output" in kw) and kw["full_output"]

    # convert x0 to internal variables
    i0 = _external2internal(x0, bounds)

    # perform unconstrained optimization using internal variables
    r = leastsq(_err, i0, args=(bounds, func, args), **kw)

    # unpack return convert to external variables and return
    if full:
        xi, cov_xi, infodic, mesg, ier = r
        xe = _internal2external(xi, bounds)
        if cov_xi:
            cov_xe = _i2e_cov_x(xi, bounds, cov_xi)
        else:
            cov_xe = None
        # XXX correct infodic 'fjac','ipvt', and 'qtf'
        return xe, cov_xe, infodic, mesg, ier

    else:
        xi, ier = r
        xe = _internal2external(xi, bounds)
        return xe, ier


def _external2internal(xe, bounds):
    """ Convert a series of external variables to internal variables

    Parameters
    ----------
    xe : array
        external variables

    bounds : list
       Same length as xe, tuples of (min, max) values for
       variable

    Returns
    -------
    array
       Variables converted to internal units
    """

    xi = np.empty_like(xe)

    for i, (v, bound) in enumerate(zip(xe, bounds)):

        a = bound[0]    # minimum
        b = bound[1]    # maximum

        if a is None and b is None:  # No constraints
            xi[i] = v

        elif b is None:     # only min
            xi[i] = np.sqrt((v-a+1.)**2.-1)

        elif a is None:     # only max
            xi[i] = np.sqrt((b-v+1.)**2.-1)

        else:   # both min and max
            xi[i] = np.arcsin((2.*(v-a)/(b-a))-1.)

    return xi


def _internal2external(xi, bounds):
    """
    Convert a series of internal variables to external variables

    Parameters
    ----------
    xi : array
       Variables in internal units

    bounds : list
       Tuples of bounds for each parameter

    Returns
    -------
    xe : array
        The variables in external units
    """

    xe = np.empty_like(xi)

    for i, (v, bound) in enumerate(zip(xi, bounds)):

        a = bound[0]    # minimum
        b = bound[1]    # maximum

        if a is None and b is None:    # No constraints
            xe[i] = v

        elif b is None:      # only min
            xe[i] = a-1.+np.sqrt(v**2.+1.)

        elif a is None:      # only max
            xe[i] = b+1.-np.sqrt(v**2.+1.)

        else:       # both min and max
            xe[i] = a+((b-a)/2.)*(np.sin(v)+1.)

    return xe


def _err(p, bounds, efunc, args):
    """
    TODO: This needs a docstring
    """

    pe = _internal2external(p, bounds)
    return efunc(pe, *args)


def _i2e_cov_x(xi, bounds, cov_x):
    """
    TODO: This needs a docstring
    """
    grad = _internal2external_grad(xi, bounds)
    grad = np.atleast_2d(grad)

    return np.dot(grad.T, grad)*cov_x


def _internal2external_grad(xi, bounds):
    """
    Calculate the internal to external gradiant
    Calculates the partial of external over internal

    TODO: This needs a completed docstring
    """

    ge = np.empty_like(xi)

    for i, (v, bound) in enumerate(zip(xi, bounds)):

        a = bound[0]    # minimum
        b = bound[1]    # maximum

        if (a is None) and (b is None):    # No constraints
            ge[i] = 1.0

        elif b is None:      # only min
            ge[i] = v/np.sqrt(v**2+1)

        elif a is None:      # only max
            ge[i] = -v/np.sqrt(v**2+1)

        else:       # both min and max
            ge[i] = (b-a)*np.cos(v)/2.

    return ge

# =============================================================================


def fit_quad_to_peak(x, y):
    """
    Fits a quadratic to the data points handed in
    to the from y = b[0](x-b[1])**2 + b[2] and R2
    (measure of goodness of fit)

    Parameters
    ----------
    x : ndarray
        locations
    y : ndarray
        values

    Returns
    -------
    b : tuple
       coefficients of form y = b[0](x-b[1])**2 + b[2]

    R2 : float
      R2 value

    """

    lenx = len(x)

    # some sanity checks
    if lenx < 3:
        raise Exception('insufficient points handed in ')
    # set up fitting array
    X = np.vstack((x ** 2, x, np.ones(lenx))).T
    # use linear least squares fitting
    beta, _, _, _ = np.linalg.lstsq(X, y)

    SSerr = np.sum(np.power(np.polyval(beta, x) - y, 2))
    SStot = np.sum(np.power(y - np.mean(y), 2))
    # re-map the returned value to match the form we want
    ret_beta = (beta[0],
                -beta[1] / (2 * beta[0]),
                beta[2] - beta[0] * (beta[1] / (2 * beta[0])) ** 2)

    return ret_beta, 1 - SSerr / SStot


def fit(x, y, param_dict, fitting_engine, target_function, limit_dict=None,
        engine_dict=None):
    """
    Top-level function for fitting, magic and ponies

    Parameters
    ----------
    x : array-like
        x-coordinates

    y : array-like
        y-coordinates

    param_dict : dict
        Dictionary of parameters

    target_function : object
        Function object that is used to compute and compare against
        y = target_function(x, **param_dict)
            x is the array of all of your data (the same parameter that was
            passed in to this function)

    fitting_engine : function_name
        Something that allows you to modify parameters to 'fit' the
        target_function to the (x,y) data
        fit_param_dict = function_name(x, y, target_function, param_dict,
                                       constraint_dict, engine_dict)

    Returns
    -------
    fit_param_dict : dict
        Dictionary of the fit values of the parameters.  Keys are the same as
        the param_dict and the limit_dict

    correlation_matrix : pandas.DataFrame
        Table of correlations (named rows/cols)

    covariance_matrix : pandas.DataFrame
        Table of covariance (named rows/cols)

    residuals : np.ndarray
        Returned as (data - fit)

    Optional
    --------
    limit_dict : dict
        Dictionary of limits for the param_dict.  Keys must be the same as the
        param_dict

    engine_dict : dict
        Dictionary of keyword arguments that the specific fitting_engine
        requires

    """

    # calls the engine

    # compute covariance

    # compute correlation

    # compute residuals

    # returns

    pass