# -*- coding: utf-8 -*-

#%%
#Basic setup
import numpy as np
from ismrmrdtools import sense
from ismrmrdtools import show
from ismrmrdtools import simulation
from ismrmrdtools import transform


#%%
reload(simulation)
matrix_size = 256
csm = simulation.generate_birdcage_sensitivities(matrix_size)
phan = simulation.phantom(matrix_size)
coil_images = np.tile(phan,(8, 1, 1)) * csm
show.imshow(abs(coil_images),tile_shape=(4,2))


#%%
#Undersample
reload(transform)
acc_factor = 2
kspace = transform.transform_image_to_kspace(coil_images,dim=(1,2))
kspace[:,1:matrix_size:acc_factor,:] = 0
noise = np.random.standard_normal(kspace.shape) + 1j*np.random.standard_normal(kspace.shape)
kspace = kspace + 2*noise/matrix_size
alias_img = transform.transform_kspace_to_image(kspace,dim=(1,2)) * np.sqrt(acc_factor)
show.imshow(abs(alias_img))

#%%
reload(sense)
(unmix, gmap) = sense.calculate_sense_unmixing(acc_factor,csm)
show.imshow(abs(gmap))

#%%
# Unalias/Combine coils
recon = np.squeeze(np.sum(alias_img * unmix,0))
show.imshow(abs(recon))
