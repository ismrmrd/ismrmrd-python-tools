"""
Tools for generating coil sensitivities and phantoms
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from ismrmrdtools import transform


def sample_data(img_obj, csm, acc=1, ref=0, sshift=0):
    """Sample the k-space of object provided in `img_obj` after first applying
    coil sensitivity maps in `csm` and Fourier transforming to k-space.

    Paramters
    ---------
    img_obj : (y, x) array
        Object in image space
    csm : (c, y, x) array
        Coil sensitivity maps
    acc : float, optional
        Acceleration factor
    ref : float, optional
        Reference lines (in center of k-space)
    sshift : float, optional
        Sampling shift, i.e for undersampling, do we start with line 1 or line
        1+sshift?.

    Returns
    -------
    data : (c, ky, kx) array
        Sampled data in k-space (zeros where not sampled).
    pat : (c, ky, kx) array
        Sampling pattern : (0 = not sampled,
                            1 = imaging data,
                            2 = reference data,
                            3 = reference and imaging data)

    Notes
    -----
    Code made available for the ISMRM 2013 Sunrise Educational Course

    Michael S. Hansen (michael.hansen@nih.gov)
    """

    sshift = sshift % acc

    if img_obj.ndim != 2:
        raise ValueError("Only two dimensional objects supported at the "
                         "moment")
    if csm.ndim != 3:
        raise ValueError("csm must be a 3 dimensional array")
    if img_obj.shape[0:2] != csm.shape[1:3]:
        raise ValueError("Object and csm dimension mismatch")

    pat_img = np.zeros(img_obj.shape, dtype=np.int8)
    pat_img[sshift:-1:acc, :] = 1
    pat_ref = np.zeros(img_obj.shape, dtype=np.int8)
    if ref > 0:
        pat_ref[(0+img_obj.shape[0]/2):(ref+img_obj.shape[0]/2), :] = 2

    pat = pat_img + pat_ref

    coil_images = np.tile(img_obj, (csm.shape[0], 1, 1)) * csm
    data = transform.transform_image_to_kspace(coil_images, dim=(1, 2))
    data = data * (np.tile(pat, (csm.shape[0], 1, 1)) > 0).astype('float32')
    return (data, pat)


def generate_birdcage_sensitivities(matrix_size=256, number_of_coils=8,
                                    relative_radius=1.5, normalize=True):
    """ Generate birdcage coil sensitivites.


    Parameters
    ----------
    matrix_size : int, optional
        size of imaging matrix in pixels.
    number_of_coils : int, optional
        Number of simulated coils.
    relative_radius : int, optional
        Relative radius of birdcage.
    normalize : bool, optional
        If True, normalize by the root sum-of-squares intensity.

    Returns
    -------
    out : array
        coil sensitivies (number_of_coils, matrix_size, matrix_size)

    This function is heavily inspired by the mri_birdcage.m Matlab script in
    Jeff Fessler's IRT package: http://web.eecs.umich.edu/~fessler/code/
    """

    out = np.zeros(
        (number_of_coils, matrix_size, matrix_size), dtype=np.complex64)
    for c in range(number_of_coils):
        coilx = relative_radius*np.cos(c*(2*np.pi/number_of_coils))
        coily = relative_radius*np.sin(c*(2*np.pi/number_of_coils))
        coil_phase = -c*(2*np.pi/number_of_coils)

        for y in range(matrix_size):
            y_co = float(y-matrix_size/2)/float(matrix_size/2)-coily
            for x in range(matrix_size):
                x_co = float(x-matrix_size/2)/float(matrix_size/2)-coilx
                rr = np.sqrt(x_co**2+y_co**2)
                phi = np.arctan2(x_co, -y_co) + coil_phase
                out[c, y, x] = (1/rr) * np.exp(1j*phi)

    if normalize:
        rss = np.squeeze(np.sqrt(np.sum(abs(out) ** 2, 0)))
        out = out / np.tile(rss, (number_of_coils, 1, 1))

    return out


def phantom(matrix_size=256, phantom_type='Modified Shepp-Logan',
            ellipses=None):
    """
    Create a Shepp-Logan [1]_ or modified Shepp-Logan [2]_ phantom.

    Parameters
    ----------
    matrix_size : int, optional
        size of imaging matrix in pixels.

    phantom_type : {'Modified Shepp-Logan', 'Shepp-Logan'}, optional
        The type of phantom to produce.  This is overridden if `ellipses` is
        also specified.

    ellipses : list or None, optional
        Custom set of ellipses to use.  See notes below for details.

    Returns
    -------
    ph : array
        Phantom image.

    Notes
    -----
    The `ellipses` should be in the form::

        [[I, a, b, x0, y0, phi],
         [I, a, b, x0, y0, phi],
                           ...]
    where each row defines an ellipse.

    I: Additive intensity of the ellipse.
    a: Length of the major axis.
    b: Length of the minor axis.
    x0: Horizontal offset of the centre of the ellipse.
    y0: Vertical offset of the centre of the ellipse.
    phi: Counterclockwise rotation of the ellipse in degrees,
         measured as the angle between the horizontal axis and
         the ellipse major axis.

    The image bounding box in the algorithm is `[-1, -1], [1, 1]`,
    so the values of `a`, `b`, `x0`, `y0` should all be specified
    with respect to this box.

    References
    ----------

    .. [1] Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
        from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
        Feb. 1974, p. 232.

    .. [2] Toft, P.; "The Radon Transform - Theory and Implementation",
        Ph.D. thesis, Department of Mathematical Modelling, Technical
        University of Denmark, June 1996.
    """

    if (ellipses is None):
        ellipses = _select_phantom(phantom_type)
    elif (np.size(ellipses, 1) != 6):
        raise ValueError("Wrong number of columns in user phantom")

    ph = np.zeros((matrix_size, matrix_size), dtype=np.float32)

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1:1:(1j*matrix_size), -1:1:(1j*matrix_size)]

    for ellip in ellipses:
        I = ellip[0]
        a2 = ellip[1]**2
        b2 = ellip[2]**2
        x0 = ellip[3]
        y0 = ellip[4]
        phi = ellip[5] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # Find the pixels within the ellipse
        locs = (((x * cos_p + y * sin_p)**2) / a2
                + ((y * cos_p - x * sin_p)**2) / b2) <= 1

        # Add the ellipse intensity to those pixels
        ph[locs] += I

    return ph


def _select_phantom(name):
    if (name.lower() == 'shepp-logan'):
        e = _shepp_logan()
    elif (name.lower() == 'modified shepp-logan'):
        e = _mod_shepp_logan()
    else:
        raise ValueError("Unknown phantom type: %s" % name)
    return e


def _shepp_logan ():
    """Standard head phantom, taken from Shepp & Logan."""
    return [[   2,   .69,   .92,    0,      0,   0],
            [-.98, .6624, .8740,    0, -.0184,   0],
            [-.02, .1100, .3100,  .22,      0, -18],
            [-.02, .1600, .4100, -.22,      0,  18],
            [ .01, .2100, .2500,    0,    .35,   0],
            [ .01, .0460, .0460,    0,     .1,   0],
            [ .02, .0460, .0460,    0,    -.1,   0],
            [ .01, .0460, .0230, -.08,  -.605,   0],
            [ .01, .0230, .0230,    0,  -.606,   0],
            [ .01, .0230, .0460,  .06,  -.605,   0]]


def _mod_shepp_logan ():
    """Modified version of Shepp & Logan's head phantom, adjusted to improve
    contrast.  Taken from Toft.
    """
    return [[   1,   .69,   .92,    0,      0,   0],
            [-.80, .6624, .8740,    0, -.0184,   0],
            [-.20, .1100, .3100,  .22,      0, -18],
            [-.20, .1600, .4100, -.22,      0,  18],
            [ .10, .2100, .2500,    0,    .35,   0],
            [ .10, .0460, .0460,    0,     .1,   0],
            [ .10, .0460, .0460,    0,    -.1,   0],
            [ .10, .0460, .0230, -.08,  -.605,   0],
            [ .10, .0230, .0230,    0,  -.606,   0],
            [ .10, .0230, .0460,  .06,  -.605,   0]]

#def ?? ():
## Add any further phantoms of interest here
#   return np.array (
#       [[ 0, 0, 0, 0, 0, 0],
#       [ 0, 0, 0, 0, 0, 0]])
