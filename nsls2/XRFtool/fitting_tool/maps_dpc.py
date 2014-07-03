"""
DPC code to calculate phase
"""




#-----------------------------------------------------------------------------
#  Applies 2-D Fourier integration to reconstruct dpc images,
#  presents fluorescence maps for select elements if present in data directory
# VARIABLE DECLARATIONS / MEANINGS:
# info                structure of info about illumination etc
# delta, beta         optical parameters
# nrml                right-minus-left normalised to transmission signal
# ntmb                top-minus-bottom normalised to transmission signal

# gxdt                 x component of the gradient of the delta.thickness
# gydt                 y component of the gradient of the delta.thickness

# ngxdt                x component of the gradient of the delta.thickness, normalised
#                         so as to have zero mean
# ngydt                y component of the gradient of the delta.thickness, normalised
#                         so as to have zero mean
#=============================================
def maps_simple_dpc_integration(nrml, ntmb, no_int = True):

    hc = 0.001239842        # wavelength-energy relationship, microns / keV


    if nrml.ndim < 1 :
        nrml = 0
        ntmb = 0
        rdt = 0
        return

    sz = nrml.shape
    nx = sz[0]
    ny = sz[1]


    # "what goes up must come down"
    #     - can be used to remove beam intensity variations AND
    #     - removes first order effect of detector misalignment
    #            (i.e., removes 'constant gradient')

    ylo = 0
    yhi = ny-1

    #find the vertical lines with the smalles
    #spread, which hopefully are the background

    for i in range(ylo, yhi):
        nrml[:, i] = nrml[:, i] - nrml[:, i].mean(axis=0)
        # added this for the other direction, too.
        ntmb[:, i] = ntmb[:, i] - ntmb[:, i].mean(axis=0)

    # remove first order effect of detector misalignment in vertical
    #            (i.e., remove 'constant gradient')
    ntmb = ntmb - ntmb.mean(axis=0)
    # added this for the other direction, too.
    nrml = nrml - nrml.mean(axis=0)

    rdt = 0

    if no_int == False:

        cs_d        = 40.0
        zp_d        = 160.0
        zp_dr        = 50.0 / 1000.
        zp_f        = 18.26 * 1000.
        energy    = 10.1

        zz        = 82.17

        hx         = 0.1
        hy         = 0.1

        xlin = np.arange(float(nx)) * hx
        ylin = np.arange(float(ny)) * hy
        ylin = ylin[::-1]
        xax = xlin # (ylin*0+1)
        yax = (xlin*0+1) # ylin

        #=============================================
        # calculate as gradient of t
        # gxdt, gydt refers to the gradient of the delta.thickness

        # extra factor of 2 comes from use of diameters, not radii...
        ngxdt = (np.math.pi * (zp_d + cs_d)) / ( 8. * zp_f) * nrml
        ngydt = (np.math.pi * (zp_d + cs_d)) / ( 8. * zp_f) * ntmb


        #=============================================
        # implement FFT reconstruction


        dpc = ngxdt + 1j * ngydt


        fx = (np.arange(float(nx)) - nx/2) / ((nx-1) * hx)
        fy = (np.arange(float(ny)) - ny/2) / ((ny-1) * hy)
        fy = fy[::-1]
        fxt = np.outer(fx,(fy*0.+1.))
        fy = np.outer((fx*0.+1.), fy )
        fx = fxt
        fxt = 0

        xy = 2j * np.math.pi * (fx +1j* fy)
        xy[(nx-1)/2, (ny-1)/2] = 1    # to avoid 0/0 error
        xy = np.fft.fftshift(xy)

        Fdpc = np.fft.fft2(dpc)

        Fdpc[0,0] = 0 # dc level information not available, leads to numerical error

        dt = np.fft.ifft2(Fdpc/xy)

        # is dt the magnitude of the complex value or the real part?
        # note that the real part dominates, and the magnitude
        # loses dynamic range due to abs(-1) = 1, i.e. real part is positive only

        idt = dt.imag
        rdt = dt.real


        temp = np.concatenate((rdt[0, 1:ny-1].flatten(), rdt[nx-1, 1:ny-1].flatten(), \
                          rdt[0:nx, 0].flatten(), rdt[0:nx, ny-1].flatten()))


        # set the average of the perimetric values to be zero
        rdt = rdt - np.mean(temp)

    return nrml, ntmb, rdt