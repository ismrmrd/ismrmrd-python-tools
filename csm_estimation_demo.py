# -*- coding: utf-8 -*-

#%%
#Basic setup
import numpy as np
from ismrmrdtools import simulation, coils, show

matrix_size = 256
csm = simulation.generate_birdcage_sensitivities(matrix_size)
phan = simulation.phantom(matrix_size)
coil_images = np.tile(phan,(8, 1, 1)) * csm
show.imshow(abs(coil_images),tile_shape=(4,2))

(csm_est, rho) = coils.calculate_csm_walsh(coil_images)
combined_image = np.sum(csm_est * coil_images, axis=0)

show.imshow(abs(csm_est),tile_shape=(4,2),scale=(0,1))
show.imshow(abs(combined_image),scale=(0,1))
