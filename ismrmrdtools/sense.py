# -*- coding: utf-8 -*-
"""
Calculate SENSE unmixing coefficients
"""
import numpy as np

def calculate_sense_unmixing(acc_factor, csm, regularization_factor = 0.001):
    '''Calculates the unmixing coefficients for a 2D image using a SENSE algorithm

    :param acc_factor: Acceleration factor, e.g. 2
    :param csm: Coil sensitivity map, ``[coil, y, x]``
    :param regularization_factor: adds tychonov regularization (default ``0.001``)

        - 0 = no regularization
        - set higher for more aggressive regularization.

    :returns unmix: Image unmixing coefficients for a single ``x`` location, ``[coil, y, x]``
    :returns gmap: Noise enhancement map, ``[y, x]``
    '''

    assert csm.ndim == 3, "Coil sensitivity map must have exactly 3 dimensions"

    unmix = np.zeros(csm.shape,np.complex64)

    for x in range(0,csm.shape[2]):
        unmix[:,:,x] = _calculate_sense_unmixing_1d(acc_factor, np.squeeze(csm[:,:,x]), regularization_factor)

    gmap = np.squeeze(np.sqrt(np.sum(abs(unmix) ** 2, 0))) * np.squeeze(np.sqrt(np.sum(abs(csm) ** 2, 0)))

    return (unmix,gmap)


def _calculate_sense_unmixing_1d(acc_factor, csm1d, regularization_factor):
    nc = csm1d.shape[0]
    ny = csm1d.shape[1]

    assert (ny % acc_factor) == 0, "ny must be a multiple of the acceleration factor"

    unmix1d = np.zeros((nc,ny),dtype=np.complex64)

    nblocks = ny/acc_factor
    for b in range(0,nblocks):
        A = np.matrix(csm1d[:,b:ny:nblocks]).T
        if np.max(np.abs(A)) > 0:
#            unmix1d[:,b:ny:nblocks] = np.linalg.pinv(A)
            AHA = A.H * A;
            reduced_eye = np.diag(np.abs(np.diag(AHA))>0)
            n_alias = np.sum(reduced_eye)
            scaled_reg_factor = regularization_factor * np.trace(AHA)/n_alias
            unmix1d[:,b:ny:nblocks] = np.linalg.pinv(AHA + (reduced_eye*scaled_reg_factor)) * A.H
    return unmix1d
