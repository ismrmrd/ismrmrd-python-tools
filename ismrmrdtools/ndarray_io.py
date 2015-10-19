from __future__ import division, print_function, absolute_import
import numpy as np
from struct import unpack


def write_ndarray(filename, ndarray):
    """Writes a simple ndarray format. This format is mostly used for debugging
    purposes.

    Paramters
    ---------
    filename : str
        Name of file containing array (extension appended automatically).
    ndarray : array
        The array to write out to `filename`.

    Notes
    -----
    The file name extension indicates the binary data format:

    '*.float' indicates float32
    '*.double' indicates float64
    '*.cplx' indicates complex64
    '*.dplx' indicatex complex128
    """
    datatype = ndarray.dtype
    if datatype == np.dtype(np.float32):
        fullfilename = filename + str('.float')
    elif datatype == np.dtype(np.float64):
        fullfilename = filename + str('.double')
    elif datatype == np.dtype(np.complex64):
        fullfilename = filename + str('.cplx')
    elif datatype == np.dtype(np.complex128):
        fullfilename = filename + str('.dplx')
    else:
        raise ValueError('Unsupported data type')

    f = open(fullfilename, 'wb')
    dims = np.zeros((ndarray.ndim+1, 1), dtype=np.int32)
    dims[0] = ndarray.ndim
    for d in range(ndarray.ndim):
        dims[d+1] = ndarray.shape[ndarray.ndim-d-1]
    f.write(dims.tobytes())
    f.write(ndarray.tobytes())


def read_ndarray(filename):
    """Read a simple ndarray format. This format is mostly used for debugging
    purposes.

    Paramters
    ---------
    filename : str
        Name of file containing array.

    Returns
    -------
    arr: array
        Numpy ndarray.

    Notes
    -----
    The file name extension indicates the binary data format:

    '*.float' indicates float32
    '*.double' indicates float64
    '*.cplx' indicates complex64
    '*.dplx' indicatex complex128
    """
    if filename.endswith('.float'):
        datatype = np.dtype(np.float32)
    elif filename.endswith('.double'):
        datatype = np.dtype(np.float64)
    elif filename.endswith('.cplx'):
        datatype = np.dtype(np.complex64)
    elif filename.endswith('.dplx'):
        datatype = np.dtype(np.complex128)
    else:
        raise Exception('Unknown file name extension')

    f = open(filename, 'rb')
    ndims = f.read(4)
    ndims = unpack('<I', ndims)[0]
    dims = np.zeros((ndims), dtype=np.int32)
    for d in range(ndims):
        di = f.read(4)
        di = unpack('<I', di)[0]
        dims[ndims-1-d] = di
    dims = tuple(dims)

    by = f.read(np.prod(dims)*datatype.itemsize)
    arr = np.frombuffer(by, datatype)
    arr = arr.reshape(dims)

    return arr
