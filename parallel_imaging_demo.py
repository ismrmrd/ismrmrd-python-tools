# -*- coding: utf-8 -*-

#%%
#Basic setup
import numpy as np
from ismrmrdtools import sense, grappa, show, simulation, transform,coils
import time

#%
#Reload some modules
reload(show)
reload(sense)
reload(grappa)
reload(simulation)
reload(transform)

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
noise = np.random.standard_normal(data.shape) + 1j*np.random.standard_normal(data.shape)
kspace = np.logical_or(pat==1,pat==3).astype('float32')*(data + (5*noise/matrix_size))
alias_img = transform.transform_kspace_to_image(kspace,dim=(1,2)) * np.sqrt(acc_factor)
show.imshow(abs(alias_img))

#%%
#Noise prewhitening
reload(coils)
#noise_tmp = noise.reshape((noise.shape[0],noise.size/noise.shape[0]))
noise_tmp = np.random.standard_normal((32,64000)) + 1j*np.random.standard_normal((32,64000))

t = time.time()
dmtx = coils.calculate_prewhitening(noise_tmp)
elapsed = time.time()-t;
print "Time Inati prewhitening: " + str(elapsed)

t = time.time()
dmtx = coils.calculate_prewhitening2(noise_tmp)
elapsed = time.time()-t;
print "Time Hansen prewhitening: " + str(elapsed)


#%%
reload(sense)
(unmix_sense, gmap_sense) = sense.calculate_sense_unmixing(acc_factor,csm)
show.imshow(abs(gmap_sense),colorbar=True)
recon_sense = np.squeeze(np.sum(alias_img * unmix_sense,0))
show.imshow(abs(recon_sense),colorbar=True)

#%%
reload(grappa)
#(unmix_grappa,gmap_grappa) = grappa.calculate_grappa_unmixing(data, acc_factor, data_mask=pat>1, csm=csm)
(unmix_grappa,gmap_grappa) = grappa.calculate_grappa_unmixing(data, acc_factor, data_mask=pat>1)
show.imshow(abs(gmap_grappa),colorbar=True)
recon_grappa = np.squeeze(np.sum(alias_img * unmix_sense,0))
show.imshow(abs(recon_grappa),colorbar=True)

