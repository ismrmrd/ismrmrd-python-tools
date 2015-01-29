# -*- coding: utf-8 -*-

#%%
#Basic setup
import numpy as np
from ismrmrdtools import sense
from ismrmrdtools import show
from ismrmrdtools import simulation
from ismrmrdtools import transform
from ismrmrdtools import grappa

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
#kspace = transform.transform_image_to_kspace(coil_images,dim=(1,2))
#kspace_full = kspace
#kspace[:,1:matrix_size:acc_factor,:] = 0
(data,pat) = simulation.sample_data(phan,csm,acc_factor,ref_lines)

noise = np.random.standard_normal(data.shape) + 1j*np.random.standard_normal(data.shape)

kspace = np.logical_or(pat==1,pat==3).astype('float32')*(data + (2*noise/matrix_size))
alias_img = transform.transform_kspace_to_image(kspace,dim=(1,2)) * np.sqrt(acc_factor)
show.imshow(abs(alias_img))

#%%
reload(sense)
(unmix, gmap) = sense.calculate_sense_unmixing(acc_factor,csm)
show.imshow(abs(gmap))

#%%
reload(grappa)
unmix = grappa.calculate_grappa_unmixing(kspace, acc_factor, data_mask=pat>1, csm=csm)
#show.imshow(abs(gmap))


#%%
# Unalias/Combine coils
recon = np.squeeze(np.sum(alias_img * unmix,0))
show.imshow(abs(recon))
