# -*- coding: utf-8 -*-

#%%
#Basic setup
import numpy as np
from ismrmrdtools import sense
from ismrmrdtools import show
from ismrmrdtools import simulation

#%%
#Loading matlab data
#exercise_data = sp.io.loadmat('hansen_exercises.mat')
#csm = exercise_data['smaps']
#csm = csm.transpose()
#csm = np.ascontiguousarray(csm)


#%%
#Show a coil sensitivity map
#show.imshow(abs(csm))


#%%
reload(simulation)
matrix_size = 256
csm = simulation.generate_birdcage_sensitivities(matrix_size)
phan = simulation.phantom(matrix_size)
coil_images = np.tile(phan,(8, 1, 1)) * csm
show.imshow(abs(coil_images),tile_shape=(4,2))

#%%
reload(sense)
(unmix, gmap) = sense.calculate_sense_unmixing(2,csm)
show.imshow(abs(gmap),scale=(1,2))
