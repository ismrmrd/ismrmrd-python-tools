import numpy as np
from numpy.fft import fftshift, ifftshift,ifftn
import coils

def calculate_grappa_unmixing(source_data, acc_factor, kernel_size=(4,5), data_mask=None, csm=None, regularization_factor=0.001, target_data=None):
    '''Calculates unmixing coefficients for a 2D image using a GRAPPA algorithm

    :param source_data: k-space source data ``[coils, y, x]``
    :param acc_factor: Acceleration factor, e.g. 2
    :param kernel_shape: Shape of the k-space kernel ``(ky-lines, kx-points)`` (default ``(4,5)``)
    :param data_mask: Mask of where calibration data is located in source_data (defaults to all of source_data)
    :param csm: Coil sensitivity map, ``[coil, y, x]`` (used for b1-weighted combining. Will be estimated from calibratino data if not supplied)
    :param regularization_factor: adds tychonov regularization (default ``0.001``)
        - 0 = no regularization
        - set higher for more aggressive regularization.
    :param target_data: If target data differs from source data (defaults to source_data)
    
    :returns unmix: Image unmixing coefficients for a single ``x`` location, ``[coil, y, x]``
    :returns gmap: Noise enhancement map, ``[y, x]``
    '''

    nx = source_data.shape[2]
    ny = source_data.shape[1]
    nc_source = source_data.shape[0]

    
    if target_data is None:
        target_data = source_data
        
    if data_mask is None:
        data_mask = np.ones((ny, nx))
        
    nc_target = target_data.shape[0]
        
    if csm is None:
        #Assume calibration data is in the middle         
        f = np.asarray(np.asmatrix(np.hamming(np.max(np.sum(data_mask,0)))).T * np.asmatrix(np.hamming(np.max(np.sum(data_mask,1)))))
        fmask = np.zeros((source_data.shape[1],source_data.shape[2]),dtype=np.complex64)
        idx = np.argwhere(data_mask==1)
        fmask[idx[:,0],idx[:,1]] = f.reshape(idx.shape[0])
        fmask = np.tile(fmask[None,:,:],(nc_source,1,1))
        csm = fftshift(ifftn(ifftshift(source_data * fmask, axes=(1,2)), axes=(1,2)), axes=(1,2))
        (csm,rho) = coils.calculate_csm_walsh(csm)
        
    
    kernel = np.zeros((nc_target,nc_source,kernel_size[0]*acc_factor,kernel_size[1]),dtype=np.complex64)
    sampled_indices = np.nonzero(data_mask)
    kx_cal = (sampled_indices[1][0],sampled_indices[1][-1])
    ky_cal = (sampled_indices[0][0],sampled_indices[0][-1])
    
    for s in range(0,acc_factor):
        kernel_mask = np.zeros((kernel_size[0]*acc_factor, kernel_size[1]),dtype=np.int8)
        kernel_mask[s:kernel_mask.shape[0]:acc_factor,:] = 1
        s_data = source_data[:,ky_cal[0]:ky_cal[1],kx_cal[0]:kx_cal[1]]
        t_data = target_data[:,ky_cal[0]:ky_cal[1],kx_cal[0]:kx_cal[1]]
        k = estimate_convolution_kernel(s_data,kernel_mask,regularization_factor=regularization_factor,target_data=t_data)
        kernel = kernel + k

    #return kernel
    
    kernel = kernel[:,:,::-1,::-1] #flip kernel in preparation for convolution
    
    csm_ss = np.sum(csm * np.conj(csm),0)
    csm_ss = csm_ss + 1.0*(csm_ss < np.spacing(1)).astype('float32')
    
    unmix = np.zeros(source_data.shape,dtype=np.complex64)
    
    for c in range(0,nc_target):
        kernel_pad = _pad_kernel(kernel[c,:,:,:],unmix.shape)
        kernel_pad = fftshift(ifftn(ifftshift(kernel_pad, axes=(1,2)), axes=(1,2)), axes=(1,2))
        kernel_pad *= unmix.shape[1]*unmix.shape[2]
        unmix = unmix + (kernel_pad * np.tile(np.conj(csm[c,:,:]) /csm_ss,(nc_source,1,1)))

    unmix /= acc_factor
    gmap = np.squeeze(np.sqrt(np.sum(abs(unmix) ** 2, 0))) * np.squeeze(np.sqrt(np.sum(abs(csm) ** 2, 0)))
    
    
    return (unmix.astype('complex64'),gmap.astype('float32'))
    
def estimate_convolution_kernel(source_data, kernel_mask, regularization_factor=0.001, target_data=None):
    '''Estimates a 2D k-space convolution kernel (as used in GRAPPA or SPIRiT)

    :param source_data: k-space source data ``[coils, y, x]``
    :param kernel_mask: Mask indicating which k-space samples to use in the neighborhood. ``[ky kx]``
    :param csm: Coil sensitivity map, ``[coil, y, x]`` (used for b1-weighted combining. Will be estimated from calibratino data if not supplied)
    :param regularization_factor: adds tychonov regularization (default ``0.001``)
        - 0 = no regularization
        - set higher for more aggressive regularization.
    :param target_data: If target data differs from source data (defaults to source_data)

    :returns unmix: Image unmixing coefficients for a single ``x`` location, ``[coil, y, x]``
    :returns gmap: Noise enhancement map, ``[y, x]``
    '''


    if target_data is None:
        target_data = source_data

    assert source_data.ndim == 3, "Source data must have exactly 3 dimensions"
    assert target_data.ndim == 3, "Targe data must have exactly 3 dimensions"
    assert kernel_mask.ndim == 2, "Kernel mask must have exactly 2 dimensions"
         
    nc_source = source_data.shape[0]
    nc_target = target_data.shape[0]

    offsets = np.argwhere(kernel_mask==1)
    offsets[:,0] -= kernel_mask.shape[0]/2
    offsets[:,1] -= kernel_mask.shape[1]/2  
    ky_range = (0-np.min(offsets[:,0]),source_data.shape[1]-np.max(offsets[:,0]))
    kx_range = (0-np.min(offsets[:,1]),source_data.shape[2]-np.max(offsets[:,1]))
    
    equations = (ky_range[1]-ky_range[0])*(kx_range[1]-kx_range[0])
    unknowns = offsets.shape[0]*nc_source    
    
    
    A = np.asmatrix(np.zeros((equations, unknowns),dtype=np.complex128))
    b = np.asmatrix(np.zeros((equations, nc_target), dtype=np.complex128))

    
    for sc in range(0,nc_source):
        for p in range(0,offsets.shape[0]):
            A[:,sc*offsets.shape[0]+p] = source_data[sc,(ky_range[0]+offsets[p,0]):(ky_range[1]+offsets[p,0]),(kx_range[0]+offsets[p,1]):(kx_range[1]+offsets[p,1])].reshape((equations,1))                       
    for tc in range(0,nc_target):
        b[:,tc] = target_data[tc,ky_range[0]:ky_range[1],kx_range[0]:kx_range[1]].reshape((equations,1))    
    
    
    if A.shape[0] < 3*A.shape[1]:
        print("Warning: number of samples in calibration data might be insufficient")
    

    S = np.linalg.svd(A,compute_uv=False)
    A_inv = np.dot(np.linalg.pinv(np.dot(A.H,A) + np.eye(A.shape[1])*(regularization_factor*np.max(np.abs(S)))**2),A.H)
    x = np.dot(A_inv,b)
    
    offsets = np.argwhere(kernel_mask==1)
    kernel = np.zeros((nc_target,nc_source, kernel_mask.shape[0],kernel_mask.shape[1]),dtype=np.complex64)
    for tc in range(0,nc_target):
        for sc in range(0,nc_source):
            for p in range(0,offsets.shape[0]):
                kernel[tc,sc,offsets[p,0],offsets[p,1]] = x[sc*offsets.shape[0]+p,tc]
    
    return kernel

def _pad_kernel(gkernel,padded_shape):
    assert gkernel.ndim == 3, "Kernel padding routine must take 3 dimensional kernel"
    padded_kernel = np.zeros(padded_shape,dtype=np.complex64)
    padding = np.asarray(padded_shape)-np.asarray(gkernel.shape)
    padding_before = (padding+1)/2
    padded_kernel[padding_before[0]:(padding_before[0]+gkernel.shape[0]),padding_before[1]:(padding_before[1]+gkernel.shape[1]),padding_before[2]:(padding_before[2]+gkernel.shape[2])] = gkernel
    return padded_kernel
     
