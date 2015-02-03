# -*- coding: utf-8 -*-

#%%
#Basic setup
import numpy as np
import scipy as sp
from ismrmrdtools import sense, grappa, show, simulation, transform,coils

#%%
#import some data
exercise_data = sp.io.loadmat('hansen_exercises2.mat')
csm = np.transpose(exercise_data['smaps'])
pat = np.transpose(exercise_data['sp'])
data = np.transpose(exercise_data['data'])
kspace = np.logical_or(pat==1,pat==3).astype('float32')*(data)

acc_factor = 4
alias_img = transform.transform_kspace_to_image(kspace,dim=(1,2)) * np.sqrt(acc_factor)
show.imshow(abs(alias_img))

(unmix_grappa,gmap_grappa) = grappa.calculate_grappa_unmixing(data, acc_factor, data_mask=pat>1, csm=csm,kernel_size=(4,5))
#(unmix_grappa,gmap_grappa) = grappa.calculate_grappa_unmixing(data, acc_factor, data_mask=pat>1)
show.imshow(abs(gmap_grappa),colorbar=True)
recon_grappa = np.squeeze(np.sum(alias_img * unmix_grappa,0))
show.imshow(abs(recon_grappa),colorbar=True)

sp.io.savemat('tmp_data.mat',{'pat_py': pat,'data_py': data,'csm_py': csm,'alias_img_py':alias_img,'unmix_grappa_py':unmix_grappa})

#%%
#Reload some modules
reload(show)
reload(sense)
reload(grappa)
reload(simulation)
reload(transform)
reload(coils)

#%%
reload(simulation)
matrix_size = 256
csm = simulation.generate_birdcage_sensitivities(matrix_size)
phan = simulation.phantom(matrix_size)
coil_images = np.tile(phan,(8, 1, 1)) * csm
show.imshow(abs(coil_images),tile_shape=(4,2),colorbar=True)

#%%
#Undersample
reload(simulation)
acc_factor = 2
ref_lines = 16
(data,pat) = simulation.sample_data(phan,csm,acc_factor,ref_lines)

#%%
#Add noise
noise = np.random.standard_normal(data.shape) + 1j*np.random.standard_normal(data.shape)
noise = (5.0/matrix_size)*noise
kspace = np.logical_or(pat==1,pat==3).astype('float32')*(data + noise)
data = (pat>0).astype('float32')*(data + noise)

#%%
#Calculate the noise prewhitening matrix
dmtx = coils.calculate_prewhitening(noise)

#%%
# Apply prewhitening
kspace = coils.apply_prewhitening(kspace, dmtx) 
data = coils.apply_prewhitening(data, dmtx) 


#%%
#Reconstruct aliased images
alias_img = transform.transform_kspace_to_image(kspace,dim=(1,2)) * np.sqrt(acc_factor)
show.imshow(abs(alias_img))


#%%
reload(sense)
(unmix_sense, gmap_sense) = sense.calculate_sense_unmixing(acc_factor,csm)
show.imshow(abs(gmap_sense),colorbar=True)
recon_sense = np.squeeze(np.sum(alias_img * unmix_sense,0))
show.imshow(abs(recon_sense),colorbar=True)

#%%
reload(grappa)
#(unmix_grappa,gmap_grappa) = grappa.calculate_grappa_unmixing(data, acc_factor, data_mask=pat>1, csm=csm,kernel_size=(2,5))
(unmix_grappa,gmap_grappa) = grappa.calculate_grappa_unmixing(data, acc_factor, data_mask=pat>1,kernel_size=(2,5))
show.imshow(abs(gmap_grappa),colorbar=True)
recon_grappa = np.squeeze(np.sum(alias_img * unmix_grappa,0))
show.imshow(abs(recon_grappa),colorbar=True)

#%% 
#Pseudo replica example
reps = 255
reps_sense = np.zeros((reps,recon_grappa.shape[0],recon_grappa.shape[1]),dtype=np.complex64)
reps_grappa = np.zeros((reps,recon_grappa.shape[0],recon_grappa.shape[1]),dtype=np.complex64)
for r in range(0,reps):
    noise_r = np.random.standard_normal(kspace.shape) + 1j*np.random.standard_normal(kspace.shape)
    kspace_r = np.logical_or(pat==1,pat==3).astype('float32')*(kspace + noise_r)
    alias_img_r = transform.transform_kspace_to_image(kspace_r,dim=(1,2)) * np.sqrt(acc_factor)
    reps_sense[r,:,:] = np.squeeze(np.sum(alias_img_r * unmix_sense,0))
    reps_grappa[r,:,:] = np.squeeze(np.sum(alias_img_r * unmix_grappa,0))

std_sense = np.std(np.real(reps_sense),0)
show.imshow(abs(std_sense),colorbar=True)
std_grappa = np.std(np.real(reps_grappa),0)
show.imshow(abs(std_grappa),colorbar=True)
