# -*- coding: utf-8 -*-

#%%
#Basic setup
import time
import numpy as np
from ismrmrdtools import simulation, coils, show

matrix_size = 256
csm = simulation.generate_birdcage_sensitivities(matrix_size)
phan = simulation.phantom(matrix_size)
coil_images = phan[np.newaxis, :, :] * csm
show.imshow(abs(coil_images), tile_shape=(4, 2))

tstart = time.time()
(csm_est, rho) = coils.calculate_csm_walsh(coil_images)
print("Walsh coil estimation duration: {}s".format(time.time() - tstart))
combined_image = np.sum(csm_est * coil_images, axis=0)

show.imshow(abs(csm_est), tile_shape=(4, 2), scale=(0, 1))
show.imshow(abs(combined_image), scale=(0, 1))

tstart = time.time()
(csm_est2, rho2) = coils.calculate_csm_inati_iter(coil_images)
print("Inati coil estimation duration: {}s".format(time.time() - tstart))
combined_image2 = np.sum(csm_est2 * coil_images, axis=0)

show.imshow(abs(csm_est2), tile_shape=(4, 2), scale=(0, 1))
show.imshow(abs(combined_image2), scale=(0, 1))
