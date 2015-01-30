# -*- coding: utf-8 -*-
"""
Utilities for coil sensivity maps, pre-whitening, etc
"""
import numpy as np
from scipy import ndimage

def calculate_prewhitening(noise):
    '''Calculates the noise pre-whitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, nsamples]``

    :returns w: Prewhitening matrix, ``[coil, coil]``, w*data is prewhitened
    '''

    assert noise.ndim == 2, "Noise data must have exactly  dimensions"
    assert noise.shape[1]>=noise.shape[0], "Need at least as many samples as channels"

    # Compute the economy svd
    (u,s,v) = np.linalg.svd(np.matrix(noise),full_matrices=False)
    # Scale by the number of samples (real and complex) and the singular values
    nsamp = noise.shape[1]
    w = np.sqrt((2.0 * (nsamp-1))) * np.diag(1/s) * u.H

    # Check
    # Rn = (1.0/(nsamp-1)) * np.dot(noise,noise.H)
    # wn = np.dot(w,noise)
    # Rwn = (1.0/(nsamp-1)) * np.dot(wn, wn.H)

    return w 
    

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
    Rs = np.zeros((ncoils,ncoils,ny,nx))
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:] = img[p,:,:] * np.conj(img[q,:,:])

    # Smooth the covariance
    for p in range(ncoils):
        for q in range(ncoils):
            smooth(Rs[p,q,:,:], smoothing)

    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    rho = np.zeros((ny, nx))
    csm = np.zeros((ncoils, ny, nx))
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
