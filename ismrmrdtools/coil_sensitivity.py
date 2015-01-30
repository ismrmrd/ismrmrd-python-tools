# -*- coding: utf-8 -*-
"""
Calculate coil sensivity maps
"""
import numpy as np
from scipy import ndimage

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

## function [v,d]=ismrm_eig_power(R);
## % function [v,d]=eig_power(R);
## %
## % vectorized method for calculating the dominant eigenvector based on
## % power method. Input, R, is an image of sample correlation matrices
## % where: R(y,x,:,:) are sample correlation matrices (ncoil x ncoil) for each pixel
## %
## % v is the dominant eigenvector
## % d is the dominant (maximum) eigenvalue

## %     ***************************************
## %     *  Peter Kellman  (kellman@nih.gov)   *
## %     *  Laboratory for Cardiac Energetics  *
## %     *  NIH NHLBI                          *
## %     ***************************************

## rows=size(R,1);cols=size(R,2);ncoils=size(R,3);
## N_iterations=2;
## v=ones(rows,cols,ncoils); % initialize e.v.

## d=zeros(rows,cols);
## for i=1:N_iterations
##     v=squeeze(sum(R.*repmat(v,[1 1 1 ncoils]),3));
## 	d=ismrm_rss(v);
##     d( d <= eps) = eps;
## 	v=v./repmat(d,[1 1 ncoils]);
## end

## p1=angle(conj(v(:,:,1)));
## % (optionally) normalize output to coil 1 phase
## v=v.*repmat(exp(sqrt(-1)*p1),[1 1 ncoils]);
## v=conj(v);

## return
