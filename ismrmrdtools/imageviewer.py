import numpy as np
import matplotlib as mpl
import pylab as plt
from matplotlib import widgets

import h5py
import numpy as np
import warnings

def read_ismrmrd_image_series(filename, groupname):
    f = h5py.File(filename)
    h = f[groupname + '/header']
    d = np.array(f[groupname + '/data'])
    a = f[groupname + '/attributes']
    
    max_average = np.max(h[:,'average'])
    max_slice = np.max(h[:,'slice'])
    max_contrast = np.max(h[:,'contrast'])
    max_phase = np.max(h[:,'phase'])
    max_repetition = np.max(h[:,'repetition'])
    max_set = np.max(h[:,'set'])
    
    im_array = np.zeros((max_average+1, max_slice+1, max_contrast+1,max_phase+1, max_repetition+1, max_set+1, d.shape[1], d.shape[2], d.shape[3], d.shape[4]), dtype=d.dtype)

    if d.size != im_array.size:
        warnings.warn("The input data does not match the expected size based on header information.")
    
    for i in range(0,d.shape[0]):
        iave = h[i,'average']
        islc = h[i,'slice']
        icon = h[i,'contrast']
        iphs = h[i,'phase']
        irep = h[i,'repetition']
        iset = h[i,'set']
        im_array[iave, islc, icon, iphs, irep, iset, :, :, :, :] = d[i,:,:,:,:]

    return h, im_array

class ImageViewer(object):
    data = None
    im_per_frame = 0
    rows = 0
    cols = 0
    axs = []
    ims = []
    fig = []
    cmap = 'gray'
    frame_slider = None
    contrast_slider = None

    def __init__(self, im_arr, frame_dimension = None):
        
        if len(im_arr.shape) < 2:
            raise Exception("Image viewer needs at least a two dimensional array")
        
        if len(im_arr.shape) == 2:
            self.data = im_arr.reshape((1,im_arr.shape[0],im_arr.shape[1]))

        if (frame_dimension is not None) and (frame_dimension < len(im_arr.shape)):
            self.data = np.moveaxis(im_arr,frame_dimension,0)
            self.im_per_frame = self.data.size/(self.data.shape[-1]*self.data.shape[-2]*self.data.shape[0])
            self.data = self.data.reshape((self.data.shape[0],self.im_per_frame, self.data.shape[-2],self.data.shape[-1]))
        else:
            self.data = im_arr
            self.im_per_frame = self.data.size/(self.data.shape[-1]*self.data.shape[-2])
            self.data = self.data.reshape((1,self.im_per_frame, self.data.shape[-2],self.data.shape[-1]))
        
        self.rows = int(np.floor(np.sqrt(self.im_per_frame)))
        self.cols = self.rows
        while self.cols*self.rows < self.im_per_frame:
            self.rows = self.rows+1

        self.draw_plot()
        
    def draw_plot(self):
        self.fig = plt.figure()

        gs1 = mpl.gridspec.GridSpec(self.rows, self.cols)
        gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 

        self.axs = []
        self.ims = []

        for r in range(0,self.rows):
            for c in range(0,self.cols):
                im_no = r*self.cols+c
                if im_no > 0:
                    self.axs.append(plt.subplot(gs1[im_no],sharex=self.axs[0],sharey=self.axs[0]))
                else:
                    self.axs.append(plt.subplot(gs1[im_no]))

                if im_no < self.im_per_frame:
                    im = self.axs[im_no].imshow(np.squeeze(self.data[0,im_no,:,:]), interpolation='none',cmap=plt.get_cmap(self.cmap))
                    self.ims.append(im)

                self.axs[im_no].axis('off')
        
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.25, 0.05, 0.5])
        self.fig.colorbar(im, cax=cbar_ax)
        
        
        slider_cmin_ax = self.fig.add_axes([0.85, 0.15, 0.10, 0.025])
        self.contrast_min_slider = widgets.Slider(slider_cmin_ax, label='cmin: ', color='black', valmin=0, valmax=np.max(self.data), valinit=0.0)
        self.contrast_min_slider.on_changed(lambda value: self.update_contrast((self.contrast_min_slider.val,self.contrast_max_slider.val)))

        slider_cmax_ax = self.fig.add_axes([0.85, 0.10, 0.10, 0.025])
        self.contrast_max_slider = widgets.Slider(slider_cmax_ax, label='cmax: ', color='black', valmin=0, valmax=np.max(self.data), valinit=np.max(self.data))
        self.contrast_max_slider.on_changed(lambda value: self.update_contrast((self.contrast_min_slider.val,self.contrast_max_slider.val)))

        slider_f_ax = self.fig.add_axes([0.85, 0.05, 0.10, 0.025])
        self.frame_slider = widgets.Slider(slider_f_ax, label='t: ', color='black', valmin=0, valmax=self.data.shape[0]-1,valfmt=' %d',valinit=0)
        self.frame_slider.on_changed(lambda value: self.update_plot(int(value)))
    
    def update_contrast(self, val):
        for im in self.ims:
            im.set_clim(val)


    def update_plot(self, frame=0):
        for r in range(0,self.rows):
            for c in range(0,self.cols):
                im_no = r*self.cols+c
                if im_no < self.im_per_frame:
                    self.ims[im_no].set_data(np.squeeze(self.data[frame,im_no,:,:]))

    def show(self):
        self.fig.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ISMRMRD Image Viewer", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--time_dimension', type=int, default=-1,help='Dimension to be interpreted as time (frame)')
    parser.add_argument('ismrmrd_file', help="ISMRMRD (HDF5) file")
    parser.add_argument('ismrmrd_group', help="Image group within HDF5 file")
    args = parser.parse_args()


    h, img_array = read_ismrmrd_image_series(args.ismrmrd_file, args.ismrmrd_group)

    if args.time_dimension > -1:
        v = ImageViewer(np.squeeze(img_array), args.time_dimension)
    else:
        v = ImageViewer(np.squeeze(img_array))

    v.show()
    plt.show()
    print "Returned from show"

if __name__ == "__main__":
    main()
