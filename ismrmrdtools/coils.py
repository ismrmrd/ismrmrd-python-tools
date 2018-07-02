# -*- coding: utf-8 -*-
"""
Utilities for coil sensivity maps, pre-whitening, etc
"""
import numpy as np
from scipy import ndimage


def calculate_prewhitening(noise, scale_factor=1.0):
    '''Calculates the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, nsamples]``
    :scale_factor: Applied on the noise covariance matrix. Used to
                   adjust for effective noise bandwith and difference in
                   sampling rate between noise calibration and actual measurement:
                   scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio

    :returns w: Prewhitening matrix, ``[coil, coil]``, w*data is prewhitened
    '''

    noise_int = noise.reshape((noise.shape[0], noise.size//noise.shape[0]))
    M = float(noise_int.shape[1])
    dmtx = (1/(M-1))*np.asmatrix(noise_int)*np.asmatrix(noise_int).H
    dmtx = np.linalg.inv(np.linalg.cholesky(dmtx))
    dmtx = dmtx*np.sqrt(2)*np.sqrt(scale_factor)
    return dmtx

def apply_prewhitening(data,dmtx):
    '''Apply the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, ...]``
    :param dmtx: Input noise prewhitening matrix

    :returns w_data: Prewhitened data, ``[coil, ...]``,
    '''

    s = data.shape
    return np.asarray(np.asmatrix(dmtx)*np.asmatrix(data.reshape(data.shape[0],data.size//data.shape[0]))).reshape(s)


def calculate_csm_walsh(img, smoothing=5, niter=3):
    '''Calculates the coil sensitivities for 2D data using an iterative version of the Walsh method

    :param img: Input images, ``[coil, y, x]``
    :param smoothing: Smoothing block size (default ``5``)
    :parma niter: Number of iterations for the eigenvector power method (default ``3``)

    :returns csm: Relative coil sensitivity maps, ``[coil, y, x]``
    :returns rho: Total power in the estimated coils maps, ``[y, x]``
    '''

    assert img.ndim == 3, "Coil sensitivity map must have exactly 3 dimensions"

    ncoils = img.shape[0]
    ny = img.shape[1]
    nx = img.shape[2]

    # Compute the sample covariance pointwise
    Rs = np.zeros((ncoils,ncoils,ny,nx),dtype=img.dtype)
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:] = img[p,:,:] * np.conj(img[q,:,:])

    # Smooth the covariance
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q] = smooth(Rs[p,q,:,:], smoothing)

    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    rho = np.zeros((ny, nx))
    csm = np.zeros((ncoils, ny, nx),dtype=img.dtype)
    for y in range(ny):
        for x in range(nx):
            R = Rs[:,:,y,x]
            v = np.sum(R,axis=0)
            lam = np.linalg.norm(v)
            v = v/lam

            for iter in range(niter):
                v = np.dot(R,v)
                lam = np.linalg.norm(v)
                v = v/lam

            rho[y,x] = lam
            csm[:,y,x] = v

    return (csm, rho)


def calculate_csm_inati_iter(im, smoothing=5, niter=5, thresh=1e-3,
                             verbose=False):
    """ Fast, iterative coil map estimation for 2D or 3D acquisitions.

    Parameters
    ----------
    im : ndarray
        Input images, [coil, y, x] or [coil, z, y, x].
    smoothing : int or ndarray-like
        Smoothing block size(s) for the spatial axes.
    niter : int
        Maximal number of iterations to run.
    thresh : float
        Threshold on the relative coil map change required for early
        termination of iterations.  If ``thresh=0``, the threshold check
        will be skipped and all ``niter`` iterations will be performed.
    verbose : bool
        If true, progress information will be printed out at each iteration.

    Returns
    -------
    coil_map : ndarray
        Relative coil sensitivity maps, [coil, y, x] or [coil, z, y, x].
    coil_combined : ndarray
        The coil combined image volume, [y, x] or [z, y, x].

    Notes
    -----
    The implementation corresponds to the algorithm described in [1]_ and is a
    port of Gadgetron's ``coil_map_3d_Inati_Iter`` routine.

    For non-isotropic voxels it may be desirable to use non-uniform smoothing
    kernel sizes, so a length 3 array of smoothings is also supported.

    References
    ----------
    .. [1] S Inati, MS Hansen, P Kellman.  A Fast Optimal Method for Coil
        Sensitivity Estimation and Adaptive Coil Combination for Complex
        Images.  In: ISMRM proceedings; Milan, Italy; 2014; p. 4407.
    """

    im = np.asarray(im)
    if im.ndim < 3 or im.ndim > 4:
        raise ValueError("Expected 3D [ncoils, ny, nx] or 4D "
                         " [ncoils, nz, ny, nx] input.")

    if im.ndim == 3:
        # pad to size 1 on z for 2D + coils case
        images_are_2D = True
        im = im[:, np.newaxis, :, :]
    else:
        images_are_2D = False

    # convert smoothing kernel to array
    if isinstance(smoothing, int):
        smoothing = np.asarray([smoothing, ] * 3)
    smoothing = np.asarray(smoothing)
    if smoothing.ndim > 1 or smoothing.size != 3:
        raise ValueError("smoothing should be an int or a 3-element 1D array")

    if images_are_2D:
        smoothing[2] = 1  # no smoothing along z in 2D case

    # smoothing kernel is size 1 on the coil axis
    smoothing = np.concatenate(([1, ], smoothing), axis=0)

    ncha = im.shape[0]

    try:
        # numpy >= 1.7 required for this notation
        D_sum = im.sum(axis=(1, 2, 3))
    except:
        D_sum = im.reshape(ncha, -1).sum(axis=1)

    v = 1/np.linalg.norm(D_sum)
    D_sum *= v
    R = 0

    for cha in range(ncha):
        R += np.conj(D_sum[cha]) * im[cha, ...]

    eps = np.finfo(im.real.dtype).eps * np.abs(im).mean()
    for it in range(niter):
        if verbose:
            print("Coil map estimation: iteration %d of %d" % (it+1, niter))
        if thresh > 0:
            prevR = R.copy()
        R = np.conj(R)
        coil_map = im * R[np.newaxis, ...]
        coil_map_conv = smooth(coil_map, box=smoothing)
        D = coil_map_conv * np.conj(coil_map_conv)
        R = D.sum(axis=0)
        R = np.sqrt(R) + eps
        R = 1/R
        coil_map = coil_map_conv * R[np.newaxis, ...]
        D = im * np.conj(coil_map)
        R = D.sum(axis=0)
        D = coil_map * R[np.newaxis, ...]
        try:
            # numpy >= 1.7 required for this notation
            D_sum = D.sum(axis=(1, 2, 3))
        except:
            D_sum = im.reshape(ncha, -1).sum(axis=1)
        v = 1/np.linalg.norm(D_sum)
        D_sum *= v

        imT = 0
        for cha in range(ncha):
            imT += np.conj(D_sum[cha]) * coil_map[cha, ...]
        magT = np.abs(imT) + eps
        imT /= magT
        R = R * imT
        imT = np.conj(imT)
        coil_map = coil_map * imT[np.newaxis, ...]

        if thresh > 0:
            diffR = R - prevR
            vRatio = np.linalg.norm(diffR) / np.linalg.norm(R)
            if verbose:
                print("vRatio = {}".format(vRatio))
            if vRatio < thresh:
                break

    coil_combined = (im * np.conj(coil_map)).sum(0)

    if images_are_2D:
        # remove singleton z dimension that was added for the 2D case
        coil_combined = coil_combined[0, :, :]
        coil_map = coil_map[:, 0, :, :]

    return coil_map, coil_combined


def smooth(img, box=5):
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    '''

    t_real = np.zeros(img.shape)
    t_imag = np.zeros(img.shape)

    ndimage.filters.uniform_filter(img.real,size=box,output=t_real)
    ndimage.filters.uniform_filter(img.imag,size=box,output=t_imag)

    simg = t_real + 1j*t_imag

    return simg
