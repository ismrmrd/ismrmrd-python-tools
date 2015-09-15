# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:59:18 2015

@author: Michael S. Hansen
"""

#%%
import os
import ismrmrd
import ismrmrd.xsd
import numpy as np
import scipy as sp

from ismrmrdtools import show, transform, coils, grappa, sense

#%%
#Convert data from siemens file with
#   siemens_to_ismrmrd -f meas_MID00032_FID22409_oil_gre_128_150reps_pause_alpha_10.dat -z 1 -o data_reps_noise.h5
#   siemens_to_ismrmrd -f meas_MID00032_FID22409_oil_gre_128_150reps_pause_alpha_10.dat -z 2 -o data_reps_data.h5
# Data can be found in Gadgetron integration test datasets

#filename_noise = 'data_reps_noise.h5'
#filename_data = 'data_reps_data.h5'

filename_noise =  'tpat3_noise.h5'
filename_data = 'tpat3_data.h5'



#%%
# Read the noise data
if not os.path.isfile(filename_noise):
    print("%s is not a valid file" % filename_noise)
    raise SystemExit
noise_dset = ismrmrd.Dataset(filename_noise, 'dataset', create_if_needed=False)

#%%
# Process the noise data
noise_reps = noise_dset.number_of_acquisitions()
a = noise_dset.read_acquisition(0)
noise_samples = a.number_of_samples
num_coils = a.active_channels
noise_dwell_time = a.sample_time_us

noise = np.zeros((num_coils,noise_reps*noise_samples),dtype=np.complex64)
for acqnum in range(noise_reps):
    acq = noise_dset.read_acquisition(acqnum)
    
    # TODO: Currently ignoring noise scans
    if not acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
       raise Exception("Errror: non noise scan found in noise calibration")

    noise[:,acqnum*noise_samples:acqnum*noise_samples+noise_samples] = acq.data
    
noise = noise.astype('complex64')
    
#%% Read the actual data
# Read the noise data
if not os.path.isfile(filename_data):
    print("%s is not a valid file" % filename_data)
    raise SystemExit
dset = ismrmrd.Dataset(filename_data, 'dataset', create_if_needed=False)

header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = header.encoding[0]

# Matrix size
eNx = enc.encodedSpace.matrixSize.x
eNy = enc.encodedSpace.matrixSize.y
eNz = enc.encodedSpace.matrixSize.z
rNx = enc.reconSpace.matrixSize.x
rNy = enc.reconSpace.matrixSize.y
rNz = enc.reconSpace.matrixSize.z

# Field of View
eFOVx = enc.encodedSpace.fieldOfView_mm.x
eFOVy = enc.encodedSpace.fieldOfView_mm.y
eFOVz = enc.encodedSpace.fieldOfView_mm.z
rFOVx = enc.reconSpace.fieldOfView_mm.x
rFOVy = enc.reconSpace.fieldOfView_mm.y
rFOVz = enc.reconSpace.fieldOfView_mm.z

#Parallel imaging factor
acc_factor = enc.parallelImaging.accelerationFactor.kspace_encoding_step_1

# Number of Slices, Reps, Contrasts, etc.
ncoils = header.acquisitionSystemInformation.receiverChannels
if enc.encodingLimits.slice != None:
    nslices = enc.encodingLimits.slice.maximum + 1
else:
    nslices = 1

if enc.encodingLimits.repetition != None:
    nreps = enc.encodingLimits.repetition.maximum + 1
else:
    nreps = 1

if enc.encodingLimits.contrast != None:
    ncontrasts = enc.encodingLimits.contrast.maximum + 1
else:
    ncontrasts = 1
    
# In case there are noise scans in the actual dataset, we will skip them. 
firstacq=0
for acqnum in range(dset.number_of_acquisitions()):
    acq = dset.read_acquisition(acqnum)
    
    if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
        print("Found noise scan at acq ", acqnum)
        continue
    else:
        firstacq = acqnum
        print("Imaging acquisition starts acq ", acqnum)
        break

#Calculate prewhiterner taking BWs into consideration
a = dset.read_acquisition(firstacq)
data_dwell_time = a.sample_time_us
noise_receiver_bw_ratio = 0.79
dmtx = coils.calculate_prewhitening(noise,scale_factor=(data_dwell_time/noise_dwell_time)*noise_receiver_bw_ratio)

    
#%%
# Process the actual data
all_data = np.zeros((nreps, ncontrasts, nslices, ncoils, eNz, eNy, rNx), dtype=np.complex64)

# Loop through the rest of the acquisitions and stuff
for acqnum in range(firstacq,dset.number_of_acquisitions()):
    acq = dset.read_acquisition(acqnum)

    acq_data_prw = coils.apply_prewhitening(acq.data,dmtx)

    # Remove oversampling if needed
    if eNx != rNx:
        xline = transform.transform_kspace_to_image(acq_data_prw, [1])
        x0 = (eNx - rNx) / 2
        x1 = (eNx - rNx) / 2 + rNx
        xline = xline[:,x0:x1]
        acq.resize(rNx,acq.active_channels,acq.trajectory_dimensions)
        acq.center_sample = rNx/2
        # need to use the [:] notation here to fill the data
        acq.data[:] = transform.transform_image_to_kspace(xline, [1])
  
    # Stuff into the buffer
    rep = acq.idx.repetition
    contrast = acq.idx.contrast
    slice = acq.idx.slice
    y = acq.idx.kspace_encode_step_1
    z = acq.idx.kspace_encode_step_2
    all_data[rep, contrast, slice, :, z, y, :] = acq.data

all_data = all_data.astype('complex64')

#%%
# Coil combination
coil_images = transform.transform_kspace_to_image(np.squeeze(np.mean(all_data,0)),(1,2))
(csm,rho) = coils.calculate_csm_walsh(coil_images)
csm_ss = np.sum(csm * np.conj(csm),0)
csm_ss = csm_ss + 1.0*(csm_ss < np.spacing(1)).astype('float32')

if acc_factor > 1:
    coil_data = np.squeeze(np.mean(all_data,0))
    reload(grappa)
    (unmix,gmap) = grappa.calculate_grappa_unmixing(coil_data, acc_factor)
    #(unmix,gmap) = sense.calculate_sense_unmixing(acc_factor,csm)
    show.imshow(abs(gmap),colorbar=True,scale=(1,2))
    
recon = np.zeros((nreps, ncontrasts, nslices, eNz, eNy, rNx), dtype=np.complex64)
for r in range(0,nreps):
    recon_data = transform.transform_kspace_to_image(np.squeeze(all_data[r,:,:,:,:,:,:]),(1,2))*np.sqrt(acc_factor)
    if acc_factor > 1:
        recon[r,:,:,:,:] = np.sum(unmix * recon_data,0)
    else:
        recon[r,:,:,:,:] = np.sum(np.conj(csm) * recon_data,0)
    
show.imshow(np.squeeze(np.std(np.abs(recon),0)),colorbar=True,scale=(1,2))

