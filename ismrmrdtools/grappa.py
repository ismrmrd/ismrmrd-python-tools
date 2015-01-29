import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift, fftn, ifftn
def calculate_grappa_unmixing(source_data, acc_factor, kernel_size=(4,5), data_mask=None, csm=None, target_data=None):
    
    nx = source_data.shape[2]
    ny = source_data.shape[1]
    nc_source = source_data.shape[0]

    
    if target_data==None:
        target_data = source_data
        
    if data_mask==None:
        data_mask = np.ones((ny, nc_source))
        
    if csm==None:
        print "Error, no CSM supped"
        return 0
        
    nc_target = target_data.shape[0]
        
#%If csm is not provided, we will estimate it.
#if (isempty(csm)),
#    if (verbose),
#        fprintf('Estimating coil sensitivity...');
#    end
#    %Apply some filtering to avoid ringing
#    f = hamming(max(sum(data_mask,1))) * hamming(max(sum(data_mask,2)))';
#    fmask = zeros(size(source_data));
#    fmask((1:size(f,1))+bitshift(size(source_data,1),-1)-bitshift(size(f,1),-1), ...
#          (1:size(f,2))+bitshift(size(source_data,2),-1)-bitshift(size(f,2),-1), :) = ...
#          repmat(f, [1 1 size(source_data,3)]);
#    csm = ismrm_transform_kspace_to_image(source_data .* fmask, [1 2]);
#    csm = ismrm_estimate_csm_walsh(csm); %Estimate coil sensitivity maps.
#    if (verbose),
#        fprintf('done.\n');
#    end
#end        
    
    kernel = np.zeros((nc_target,nc_source,kernel_size[0]*acc_factor,kernel_size[1]),dtype=np.complex64)
    sampled_indices = np.nonzero(data_mask)
    kx_cal = (sampled_indices[1][0],sampled_indices[1][-1])
    ky_cal = (sampled_indices[0][0],sampled_indices[0][-1])
    
    for s in range(0,acc_factor-1):
        kernel_mask = np.zeros((kernel_size[0]*acc_factor, kernel_size[1]),dtype=np.int8)
        kernel_mask[s:kernel_mask.shape[0]:acc_factor,:] = 1
        s_data = source_data[:,ky_cal[0]:ky_cal[1],kx_cal[0]:kx_cal[1]]
        t_data = target_data[:,ky_cal[0]:ky_cal[1],kx_cal[0]:kx_cal[1]]
        k = estimate_convolution_kernel(s_data,kernel_mask,t_data)
        kernel = kernel + k

    kernel = kernel[:,:,::-1,::-1] #flip kernel in preparation for convolution
    
    csm_ss = np.sum(csm * np.conj(csm),0)
    csm_ss = csm_ss + 1.0*(csm_ss < np.spacing(1)).astype('float32')
    
    unmix = np.zeros(source_data.shape,dtype=np.complex64)
    
    for c in range(0,nc_target):
        kernel_pad = pad_kernel(kernel[c,:,:,:],unmix.shape)
        kernel_pad = fftshift(ifftn(ifftshift(kernel_pad, axes=(1,2)), axes=(1,2)), axes=(1,2))
        kernel_pad *= unmix.shape[1]*unmix.shape[2]
        unmix = unmix + (kernel_pad * np.tile(np.conj(csm[c,:,:]) /csm_ss,(nc_source,1,1)))
#
#%Loop over target coils and fo b1-weighted combination in image space.
#csm_ss = sum(conj(csm).*csm,3);
#csm_ss(csm_ss < eps) = 1; %Avoid devision by zeros where coils are undefined
#for c=1:target_coils,
#    kernel_pad = pad_grappa_kernel(kernel(:,:,:,c),size(target_data));
#    kernel_pad = fftshift(ifft(ifftshift(kernel_pad,1),[],1),1);
#    kernel_pad = fftshift(ifft(ifftshift(kernel_pad,2),[],2),2);
#    kernel_pad = kernel_pad*(size(kernel_pad,1)*size(kernel_pad,2));
#    unmix = unmix + (kernel_pad .* repmat(conj(csm(:,:,c)) ./csm_ss,[1 1 coils]));
#end
#
#unmix = unmix/acc_factor;    
    
    return unmix
    
def estimate_convolution_kernel(source_data, kernel_mask, target_data=None):

    if target_data == None:
        target_data = source_data

    assert source_data.ndim == 3, "Source data must have exactly 3 dimensions"
    assert target_data.ndim == 3, "Targe data must have exactly 3 dimensions"
    assert kernel_mask.ndim == 2, "Kernel mask must have exactly 2 dimensions"
         
#    nx = source_data.shape[2]
#    ny = source_data.shape[1]
    nc_source = source_data.shape[0]
    nc_target = target_data.shape[0]

    offsets = np.argwhere(kernel_mask==1)
    offsets[:,0] -= kernel_mask.shape[0]/2
    offsets[:,1] -= kernel_mask.shape[1]/2  
    ky_range = (0-np.min(offsets[:,0]),source_data.shape[1]-np.max(offsets[:,0]))
    kx_range = (0-np.min(offsets[:,1]),source_data.shape[2]-np.max(offsets[:,1]))
    
    equations = (ky_range[1]-ky_range[0])*(kx_range[1]-kx_range[0])
    unknowns = offsets.shape[0]*nc_source    
    
    
#    print "source_data.shape: " + str(source_data.shape)
#    print "ky_range: " + str(ky_range)
#    print "kx_range: " + str(kx_range)
    
    A = np.asmatrix(np.zeros((equations, unknowns),dtype=np.complex64))
    b = np.asmatrix(np.zeros((equations, nc_target), dtype=np.complex64))


    for sc in range(0,nc_source):
        for p in range(0,offsets.shape[0]):
            A[:,sc*offsets.shape[0]+p] = source_data[sc,(ky_range[0]+offsets[p,0]):(ky_range[1]+offsets[p,0]),(kx_range[0]+offsets[p,1]):(kx_range[1]+offsets[p,1])].reshape((equations,1))                       
    for tc in range(0,nc_target):
        b[:,tc] = target_data[tc,ky_range[0]:ky_range[1],kx_range[0]:kx_range[1]].reshape((equations,1))    
    
    
    if A.shape[0] < 3*A.shape[1]:
        print "Warning: number of samples in calibration data might be insufficient"
    
    S = np.linalg.svd(A,compute_uv=False)
    A_inv = np.linalg.pinv(A.H*A + np.eye(A.shape[1])*(1e-3*np.max(abs(S))**2))*A.H
    x = A_inv*b

    kernel = np.zeros((nc_target,nc_source, kernel_mask.shape[0],kernel_mask.shape[1]),dtype=np.complex64)
    rep_mask = np.tile(kernel_mask,(nc_target,nc_source)).reshape(nc_target,nc_source,kernel_mask.shape[0],kernel_mask.shape[1])
    sampled_indices = np.argwhere(rep_mask==1)
    kernel.ravel()[np.ravel_multi_index((sampled_indices[:,0],sampled_indices[:,1],sampled_indices[:,2],sampled_indices[:,3]),rep_mask.shape)] = x.ravel()
    
    return kernel

def pad_kernel(gkernel,padded_shape):
    assert gkernel.ndim == 3, "Kernel padding routine must take 3 dimensional kernel"
    padded_kernel = np.zeros(padded_shape,dtype=np.complex64)
    padding = np.asarray(padded_shape)-np.asarray(gkernel.shape)
    padding_before = (padding+1)/2
    padded_kernel[padding_before[0]:(padding_before[0]+gkernel.shape[0]),padding_before[1]:(padding_before[1]+gkernel.shape[1]),padding_before[2]:(padding_before[2]+gkernel.shape[2])] = gkernel
    return padded_kernel
     
#function [unmix, gmap] = ismrm_calculate_grappa_unmixing(source_data, kernel_size, acc_factor, data_mask, csm, target_data, verbose)
#%
#%   [unmix, gmap] = ismrm_calculate_grappa_unmixing(source_data, kernel_size, acc_factor, data_mask, csm, target_data, verbose)
#%   
#%   Calculates b1-weighted image space GRAPPA unmixing coefficients.
#%
#%   INPUT:
#%       source_data [kx,ky,coil]   : Source data for grappa kernel estimation (k-space)
#%       kernel_size [kx,ky]        : e.g. [4 5]
#%       acc_factor  scalar         : Acceleration factor, e.g. 2
#%       data_mask   [kx,ky]        : '1' = calibration data, '0' = ignore
#%       csm         [x,y,c]        : Coil sensitivity map, if empty, it
#%                                    will be estimated from the reference lines.
#%       target_data [kx,ky,coil]   : Target coil data, defaults to source data
#%       verbose     bool           : Set true for verbose output
#%
#%   OUTPUT:
#%       unmix [x,y,coil]           : Image unmixing coefficients
#%       gmap  [x, y]               : Noise enhancement map 
#%
#%   Typical usage:
#%       [unmix] = calculate_grappa_unmixing(source_data, [5 4], 4);
#%
#%
#%   Notes:
#%     - The unmixing coefficients produced by this routine produce uniform 
#%       noise distribution images when there is no acceleration, i.e. the
#%       noise in each pixel will be input noise * g-factor, where g-factor
#%       is sqrt(sum(abs(unmix).^2,3)).
#%
#%       If you have coil sensitivities where the RSS of the coil
#%       sensitivites is not 1 in each pixel, e.g. as obtained with a
#%       seperate calibration scan using a body coil, and you would like a
#%       uniform sensitivity image. You must apply that weighting after the
#%       parallel imaging reconstruction by dividin with the RSS of the coil
#%       sensitivites. 
#%
#%   Code made available for the ISMRM 2013 Sunrise Educational Course
#% 
#%   Michael S. Hansen (michael.hansen@nih.gov)
#%
#
#if nargin < 3,
#   error('At least 4 arguments needed'); 
#end
#
#if nargin < 4,
#    data_mask = [];
#end
#
#if nargin < 5,
#    csm = [];
#end
#
#if nargin < 6,
#    target_data = [];
#end
#
#if nargin < 7,
#    verbose = false;
#end
#
#if (isempty(target_data)),
#        target_data = source_data;
#end
#
#
#if (isempty(data_mask)),
#    data_mask = ones(size(source_data,1),size(source_data,2));
#end
#
#if (length(size(source_data)) == 2),
#    coils = 1;
#else
#    coils = size(source_data,length(size(source_data)));
#end
#
#if (length(size(target_data)) == 2),
#    target_coils = 1;
#else
#    target_coils = size(target_data,length(size(target_data)));
#end
#
#%If csm is not provided, we will estimate it.
#if (isempty(csm)),
#    if (verbose),
#        fprintf('Estimating coil sensitivity...');
#    end
#    %Apply some filtering to avoid ringing
#    f = hamming(max(sum(data_mask,1))) * hamming(max(sum(data_mask,2)))';
#    fmask = zeros(size(source_data));
#    fmask((1:size(f,1))+bitshift(size(source_data,1),-1)-bitshift(size(f,1),-1), ...
#          (1:size(f,2))+bitshift(size(source_data,2),-1)-bitshift(size(f,2),-1), :) = ...
#          repmat(f, [1 1 size(source_data,3)]);
#    csm = ismrm_transform_kspace_to_image(source_data .* fmask, [1 2]);
#    csm = ismrm_estimate_csm_walsh(csm); %Estimate coil sensitivity maps.
#    if (verbose),
#        fprintf('done.\n');
#    end
#end
#    
#
#kernel = zeros(kernel_size(1),kernel_size(2)*acc_factor,coils,target_coils);
#
#if (verbose),
#    fprintf('Calculating grappa kernels...\n');
#end
#
#[kx_cal,ky_cal] = ind2sub(size(data_mask),[find(data_mask == 1,1,'first') find(data_mask == 1,1,'last')]);
#for s=1:acc_factor,
#   kernel_mask = zeros(kernel_size(1),kernel_size(2)*acc_factor);
#   kernel_mask(:,s:acc_factor:end) = 1;
#   k = ismrm_estimate_convolution_kernel(source_data(kx_cal(1):kx_cal(2),ky_cal(1):ky_cal(2),:),kernel_mask,target_data(kx_cal(1):kx_cal(2),ky_cal(1):ky_cal(2),:));
#   kernel = kernel + k;
#end
#
#
#
#kernel = flipdim(flipdim(kernel,1),2); %Flip dimensions in preparation for convolution.
#
#unmix = zeros(size(source_data));
#if (nargout > 2),
#   unmix_sc = zeros(size(unmix,1),size(unmix,2),coils,coils); 
#end
#
#if (verbose),
#    fprintf('Doing B1 weighted combination....');
#end
#
#%Loop over target coils and fo b1-weighted combination in image space.
#csm_ss = sum(conj(csm).*csm,3);
#csm_ss(csm_ss < eps) = 1; %Avoid devision by zeros where coils are undefined
#for c=1:target_coils,
#    kernel_pad = pad_grappa_kernel(kernel(:,:,:,c),size(target_data));
#    kernel_pad = fftshift(ifft(ifftshift(kernel_pad,1),[],1),1);
#    kernel_pad = fftshift(ifft(ifftshift(kernel_pad,2),[],2),2);
#    kernel_pad = kernel_pad*(size(kernel_pad,1)*size(kernel_pad,2));
#    unmix = unmix + (kernel_pad .* repmat(conj(csm(:,:,c)) ./csm_ss,[1 1 coils]));
#end
#
#unmix = unmix/acc_factor;
#
#if (nargout > 1),
#   gmap = sqrt(sum(abs(unmix).^2,3)) .* sqrt(sum(abs(csm).^2,3));
#end
#
#
#if (verbose),
#    fprintf('done.\n');
#end
#
#return
#
#%Utility function for padding grappa kernel
#function padded_kernel = pad_grappa_kernel(gkernel, image_size)
#    padded_kernel = zeros(image_size(1),image_size(2),size(gkernel,3));
#    padded_kernel([1:size(gkernel,1)]+bitshift(image_size(1)-size(gkernel,1),-1)+1, ...
#        [1:size(gkernel,2)]+bitshift(image_size(2)-size(gkernel,2),-1)+1, :) = gkernel;
#return
#        
