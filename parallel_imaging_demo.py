# -*- coding: utf-8 -*-

#%%
#Basic setup
import numpy as np
from ismrmrdtools import sense, grappa, show, simulation, transform

#%%
reload(simulation)
matrix_size = 256
csm = simulation.generate_birdcage_sensitivities(matrix_size)
phan = simulation.phantom(matrix_size)
coil_images = np.tile(phan,(8, 1, 1)) * csm
show.imshow(abs(coil_images),tile_shape=(4,2))


#%%
#Undersample
reload(simulation)
acc_factor = 2
ref_lines = 16
(data,pat) = simulation.sample_data(phan,csm,acc_factor,ref_lines)
noise = np.random.standard_normal(data.shape) + 1j*np.random.standard_normal(data.shape)
kspace = np.logical_or(pat==1,pat==3).astype('float32')*(data + (2*noise/matrix_size))
alias_img = transform.transform_kspace_to_image(kspace,dim=(1,2)) * np.sqrt(acc_factor)
show.imshow(abs(alias_img))

#%%
reload(sense)
(unmix_sense, gmap_sense) = sense.calculate_sense_unmixing(acc_factor,csm)
show.imshow(abs(gmap_sense))
recon_sense = np.squeeze(np.sum(alias_img * unmix_sense,0))
show.imshow(abs(recon_sense))

#%%
reload(grappa)
(unmix_grappa,gmap_grappa) = grappa.calculate_grappa_unmixing(data, acc_factor, data_mask=pat>1, csm=csm)
show.imshow(abs(gmap_grappa))
recon_grappa = np.squeeze(np.sum(alias_img * unmix_sense,0))
show.imshow(abs(recon_grappa))

