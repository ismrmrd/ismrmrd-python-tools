from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.fft import fftshift, ifftshift, ifftn
from . import coils


def calculate_grappa_unmixing(source_data, acc_factor, kernel_size=(4, 5),
                              data_mask=None, csm=None,
                              regularization_factor=0.001, target_data=None):
    """Calculates unmixing coefficients for a 2D image using a GRAPPA algorithm

    Paramters
    ---------
    source_data : (coils, y, x) array
        k-space source data.
    acc_factor : int
        Acceleration factor, e.g. 2
    kernel_shape : tuple, optional
        Shape of the k-space kernel (ky-lines, kx-points).
    data_mask : (y, x) array or None, optional
        Mask of where calibration data is located in source_data.  defaults to
        all of the source data.
    csm : (coil, y, x) array or None, optional
        Coil sensitivity map. (used for b1-weighted combining. Will be
        estimated from calibratino data if not supplied.)
    regularization_factor : float, optional
        Tikhonov regularization weight.
            - 0 = no regularization
            - set higher for more aggressive regularization.
    target_data : (coil, y, x) array or None, optional
        If target data differs from source data. (defaults to source_data)


    Returns
    -------
    unmix : (coil, y, x) array
        Image unmixing coefficients for a single ``x`` location.
    gmap : (y, x) array
        Noise enhancement map.
    """

    nx = source_data.shape[2]
    ny = source_data.shape[1]
    nc_source = source_data.shape[0]

    if target_data is None:
        target_data = source_data

    if data_mask is None:
        data_mask = np.ones((ny, nx))

    nc_target = target_data.shape[0]

    if csm is None:
        # Assume calibration data is in the middle
        f = np.asarray(np.asmatrix(np.hamming(np.max(
            np.sum(data_mask, 0)))).T * np.asmatrix(
                np.hamming(np.max(np.sum(data_mask, 1)))))
        fmask = np.zeros(
            (source_data.shape[1], source_data.shape[2]), dtype=np.complex64)
        idx = np.argwhere(data_mask == 1)
        fmask[idx[:, 0], idx[:, 1]] = f.reshape(idx.shape[0])
        fmask = np.tile(fmask[None, :, :], (nc_source, 1, 1))
        csm = fftshift(
            ifftn(
                ifftshift(source_data * fmask, axes=(1, 2)),
                axes=(1, 2)),
            axes=(1, 2))
        (csm, rho) = coils.calculate_csm_walsh(csm)

    kernel = np.zeros((nc_target, nc_source, kernel_size[0]*acc_factor,
                       kernel_size[1]), dtype=np.complex64)
    sampled_indices = np.nonzero(data_mask)
    kx_cal = (sampled_indices[1][0], sampled_indices[1][-1])
    ky_cal = (sampled_indices[0][0], sampled_indices[0][-1])

    for s in range(acc_factor):
        kernel_mask = np.zeros(
            (kernel_size[0]*acc_factor, kernel_size[1]), dtype=np.int8)
        kernel_mask[s:kernel_mask.shape[0]:acc_factor, :] = 1
        s_data = source_data[:, ky_cal[0]:ky_cal[1], kx_cal[0]:kx_cal[1]]
        t_data = target_data[:, ky_cal[0]:ky_cal[1], kx_cal[0]:kx_cal[1]]
        k = estimate_convolution_kernel(
            s_data, kernel_mask, regularization_factor=regularization_factor,
            target_data=t_data)
        kernel = kernel + k

    # return kernel

    # flip kernel in preparation for convolution
    kernel = kernel[:, :, ::-1, ::-1]

    csm_ss = np.sum(csm * np.conj(csm), 0)
    csm_ss = csm_ss + 1.0*(csm_ss < np.spacing(1)).astype('float32')

    unmix = np.zeros(source_data.shape, dtype=np.complex64)

    for c in range(nc_target):
        kernel_pad = _pad_kernel(kernel[c, :, :, :], unmix.shape)
        kernel_pad = fftshift(
            ifftn(ifftshift(kernel_pad, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))
        kernel_pad *= unmix.shape[1]*unmix.shape[2]
        unmix = unmix + \
            (kernel_pad *
             np.tile(np.conj(csm[c, :, :]) / csm_ss, (nc_source, 1, 1)))

    unmix /= acc_factor
    gmap = np.squeeze(np.sqrt(np.sum(abs(unmix) ** 2, 0))) * \
        np.squeeze(np.sqrt(np.sum(abs(csm) ** 2, 0)))

    return (unmix.astype('complex64'), gmap.astype('float32'))


def estimate_convolution_kernel(source_data, kernel_mask,
                                regularization_factor=0.001, target_data=None):
    """Estimates a 2D k-space convolution kernel (as used in GRAPPA or SPIRiT).

    Paramters
    ---------
    source_data : (coil, y, x) array
        k-space source data.
    kernel_mask : (ky, kx) array
        Mask indicating which k-space samples to use in the neighborhood.
    regularization_factor : float, optional
        Tikhonov regularization weight
            - 0 = no regularization
            - set higher for more aggressive regularization.
    target_data : (coil, y, x) array or None, optional
        If target data differs from source data (defaults to source_data)

    Returns
    -------
    unmix : (coil, y, x) array
        Image unmixing coefficients for a single ``x`` location.
    gmap : (y, x) array
        Noise enhancement map.
    """

    if target_data is None:
        target_data = source_data

    if source_data.ndim != 3:
        raise ValueError("Source data must have exactly 3 dimensions")
    if target_data.ndim != 3:
        raise ValueError("Targe data must have exactly 3 dimensions")
    if kernel_mask.ndim != 2:
        raise ValueError("Kernel mask must have exactly 2 dimensions")

    nc_source = source_data.shape[0]
    nc_target = target_data.shape[0]

    offsets = np.argwhere(kernel_mask == 1)
    offsets[:, 0] -= kernel_mask.shape[0]/2
    offsets[:, 1] -= kernel_mask.shape[1]/2
    ky_range = (
        0-np.min(offsets[:, 0]), source_data.shape[1]-np.max(offsets[:, 0]))
    kx_range = (
        0-np.min(offsets[:, 1]), source_data.shape[2]-np.max(offsets[:, 1]))

    equations = (ky_range[1]-ky_range[0])*(kx_range[1]-kx_range[0])
    unknowns = offsets.shape[0]*nc_source

    A = np.zeros((equations, unknowns), dtype=np.complex128)
    b = np.zeros((equations, nc_target), dtype=np.complex128)

    for sc in range(nc_source):
        for p in range(offsets.shape[0]):
            yslice = slice(ky_range[0]+offsets[p, 0],
                           ky_range[1]+offsets[p, 0])
            xslice = slice(kx_range[0]+offsets[p, 1],
                           kx_range[1]+offsets[p, 1])
            A[:, sc*offsets.shape[0]+p] = source_data[
                sc, yslice, xslice].reshape((equations, ))
    for tc in range(nc_target):
        b[:, tc] = target_data[
            tc, ky_range[0]:ky_range[1],
            kx_range[0]:kx_range[1]].reshape((equations, ))

    if A.shape[0] < 3*A.shape[1]:
        print("Warning: number of samples in calibration data might be "
              "insufficient")

    S = np.linalg.svd(A, compute_uv=False)
    A_inv = np.dot(np.linalg.pinv(np.dot(np.conj(A.T), A) + np.eye(A.shape[1]) *
                   (regularization_factor*np.max(np.abs(S)))**2), np.conj(A.T))
    x = np.dot(A_inv, b)

    offsets = np.argwhere(kernel_mask == 1)
    kernel = np.zeros((nc_target, nc_source, kernel_mask.shape[
                      0], kernel_mask.shape[1]), dtype=np.complex64)
    for tc in range(nc_target):
        for sc in range(nc_source):
            for p in range(offsets.shape[0]):
                kernel[tc, sc, offsets[p, 0], offsets[p, 1]] = x[
                    sc*offsets.shape[0]+p, tc]

    return kernel


def _pad_kernel(gkernel, padded_shape):
    if gkernel.ndim != 3:
        raise ValueError("Kernel padding routine must take 3 dimensional "
                         "kernel")
    padded_kernel = np.zeros(padded_shape, dtype=np.complex64)
    padding = np.asarray(padded_shape)-np.asarray(gkernel.shape)
    padding_before = (padding+1)/2
    pad_slices = [
        slice(padding_before[0], padding_before[0]+gkernel.shape[0]),
        slice(padding_before[1], padding_before[1]+gkernel.shape[1]),
        slice(padding_before[2], padding_before[2]+gkernel.shape[2])]
    padded_kernel[pad_slices] = gkernel
    return padded_kernel
