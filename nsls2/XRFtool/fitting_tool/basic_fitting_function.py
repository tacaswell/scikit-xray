"""
basic fitting functions used for Fluorescence fitting
"""
import numpy as np
import scipy.special
import scipy.signal

#-----------------------------------------------------------------------------
def erf(x):
    # save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

def erfc(x):
    return 1-erf(x)


def model_gauss_peak(gain, sigma, delta_energy):
    """
    models a gaussian fluorescence peak 
    please refer to van espen, spectrum evaluation in van grieken, 
    handbook of x-ray spectrometry, 2nd ed, page 182 ff
    """
    
    counts = gain / ( sigma *np.sqrt(2.*np.math.pi)) * np.exp( -0.5* ((delta_energy / sigma)**2) )

    return counts



def model_gauss_step(gain, sigma, delta_energy, peak_E):
    """
    models a gaussian step fluorescence peak, 
    please refer to van espen, spectrum evaluation,
    in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182 ff
    """
    
    counts = gain / 2. /  peak_E * scipy.special.erfc(delta_energy/(np.sqrt(2)*sigma))

    return counts




def model_gauss_tail(gain, sigma, delta_energy, gamma):
    """
    models a gaussian fluorescence peak
    please see also van espen, spectrum evaluation,
    in van grieken, handbook of x-ray spectrometry, 2nd ed, page 182 ff
    """

    delta_energy_neg = delta_energy.copy()
    #wo_neg = np.where(delta_energy_neg > 0.)
    wo_neg = (np.nonzero(delta_energy_neg > 0.))[0]
    if wo_neg.size > 0:
        delta_energy_neg[wo_neg] = 0.
    temp_a = np.exp(delta_energy_neg/ (gamma * sigma))
    counts = gain / 2. / gamma / sigma / np.exp(-0.5/(gamma**2)) *  \
        temp_a * scipy.special.erfc( delta_energy  /( np.sqrt(2)*sigma) + (1./(gamma*np.sqrt(2)) )  )


    return counts



def elastic_peak(fitp, counts, ev, p, gain, matrix = False):
    """
    elastic peak as a gaussian function
    """
    keywords = fitp.keywords
    sigma = np.sqrt( (p[keywords.fwhm_pos[0]]/2.3548)**2  +  (p[keywords.coherent_pos[0]])*2.96*p[keywords.fwhm_pos[1]]  )
    delta_energy = ev - (p[keywords.coherent_pos[0]])

    # elastic peak, gaussian
    value = 1.
    if matrix == False :
        value = value * 10.**(p[keywords.coherent_pos[1]])
    value = value * model_gauss_peak(gain, sigma, delta_energy)
    counts = counts + value

    return counts, sigma




def compton_peak(fitp, counts, ev, p, gain, matrix = False):

    keywords = fitp.keywords
    compton_E = p[keywords.coherent_pos[0]]/(1. +(p[keywords.coherent_pos[0]]/511.)*(1. -np.cos( p[keywords.compton_pos[0]]*2.*np.math.pi/360. )))

    sigma = np.sqrt( (p[keywords.fwhm_pos[0]]/2.3548)**2 + compton_E*2.96*p[keywords.fwhm_pos[1]]  )

    local_sigma = sigma*p[14]

    delta_energy = ev.copy() - compton_E

    # compton peak, gaussian
    faktor = 1. / (1. +p[keywords.compton_pos[3]]+p[keywords.compton_pos[4]]+p[keywords.compton_pos[6]])
    if matrix == False :
        faktor = faktor * (10.**p[keywords.compton_pos[2]])
    value = faktor * model_gauss_peak(gain, sigma*p[keywords.compton_pos[1]], delta_energy)
    counts = counts + value

    # compton peak, step
    if p[keywords.compton_pos[3]] > 0.:
        value = faktor * p[keywords.compton_pos[3]]
        value = value * model_gauss_step(gain, sigma, delta_energy, compton_E)
        counts = counts + value

    # compton peak, tail on the low side
    value = faktor * p[keywords.compton_pos[4]]
    value = value * model_gauss_tail(gain, sigma, delta_energy, p[keywords.compton_pos[5]])
    counts = counts + value

    # compton peak, tail on the high side
    value = faktor * p[keywords.compton_pos[6]]
    value = value * model_gauss_tail(gain, sigma, -1.*delta_energy, p[keywords.compton_pos[7]])
    counts = counts + value

    return counts, sigma, faktor
    
    
    
    
def maps_snip(fitp, par_values):
    """
    generate background
    """
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

    energy = e_off + energy * e_lin + np.power(energy,2) * e_quad


    tmp = (e_off/2.3548)**2 + energy*2.96*e_lin
    wind = np.nonzero(tmp < 0)[0]
    tmp[wind] = 0.
    fwhm = 2.35 * np.sqrt(tmp)


    original_bcgrd = background.copy()

    #import matplotlib.pyplot as plt
    #plt.plot(energy, background)
    #plt.show()

    #smooth the background
    if keywords.spectral_binning > 0 :
        s = scipy.signal.boxcar(3)
    else :
        s = scipy.signal.boxcar(5)
    A = s.sum()
    background = scipy.signal.convolve(background,s,mode='same')/A

    #Check smoothing
    #plt.plot(energy, background)
    #plt.show()

    # SNIP PARAMETERS
    window_rf = np.sqrt(2)

    width = par_values[keywords.added_params[0]]

    window_p = width * fwhm / e_lin # in channels
    if keywords.spectral_binning > 0:
        window_p = window_p/2.

    #print "#########window_p: ", window_p
    background = np.log(np.log(background+1.)+1.)

    index = np.arange(np.float(n_background))
    #print "########index ", index
    #FIRST SNIPPING

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

        temp = (background[lo_index.astype(np.int)] + background[hi_index.astype(np.int)]) / 2.
        wo = np.where(background > temp)
        background[wo] = temp[wo]

    #import matplotlib.pyplot as plt
    #plt.plot(energy, np.exp(np.exp(background)-1.)-1.)
    #plt.show()

    if keywords.spectral_binning > 0:
        no_iterations = 7
    else:
        no_iterations = 12

    current_width = window_p
    max_current_width = np.amax(current_width)

    while max_current_width >= 0.5:
        #print 'max_current_width = ', max_current_width
        lo_index = index - current_width
        wo = np.where(lo_index < max((keywords.xmin, 0)))
        lo_index[wo] = max((keywords.xmin, 0))
        hi_index = index + current_width
        wo = np.where(hi_index > min((keywords.xmax, n_background-1)))
        hi_index[wo] = min((keywords.xmax, n_background-1))

        temp = (background[lo_index.astype(np.int)] + background[hi_index.astype(np.int)]) / 2.
        wo = np.where(background > temp)
        background[wo] = temp[wo]

        current_width = current_width / window_rf
        max_current_width = np.amax(current_width)

    #import matplotlib.pyplot as plt
    #plt.plot(energy, np.exp(np.exp(background)-1.)-1.)
    #plt.show()

    background = np.exp(np.exp(background)-1.)-1.

    wo = np.where(np.isfinite(background) == False)
    background[wo] = 0.

    keywords.background = background.copy()


    return background




    
    
    



    
    
    
    