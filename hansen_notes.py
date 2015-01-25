# -*- coding: utf-8 -*-

#%%
#Basic setup
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import sense

#%%
#Loading matlab data
exercise_data = sp.io.loadmat('hansen_exercises.mat')
csm = exercise_data['smaps']
csm = csm.transpose()
csm = np.ascontiguousarray(csm)


#%%
#Show a coil sensitivity map
plt.imshow(abs(csm[1,:,:]))

#%%
reload(sense)
(unmix, gmap) = sense.calculate_sense_unmixing(4,csm)
plt.imshow(abs(gmap))
