"""
Helpers for transforming data from k-space to image space and vice-versa.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn


def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Compute the Fourier transform from k-space to image space
    along a given or all dimensions.

    Paramters
    ---------
    k : array
        k-space data
    dim : tuple, optional
        vector of dimensions to transform
    img_shape : tuple, optional
        desired shape of output image

    Returns
    -------
    img : array
        data in image space (along transformed dimensions)
    """
    if dim is None:
        dim = tuple(range(k.ndim))

    img = fftshift(
        ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """Compute the Fourier transform from image space to k-space space
    along a given or all dimensions.

    Paramters
    ---------
    img : array
        image space data
    dim : tuple, optional
        vector of dimensions to transform
    k_shape : tuple, optional
        desired shape of output k-space data

    Returns
    -------
    k : array
        data in k-space (along transformed dimensions)
    """
    if dim is None:
        dim = tuple(range(img.ndim))

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k
