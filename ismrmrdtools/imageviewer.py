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

class WindowLevelMouse:

    def __init__(self, viewer):
        self.press = None
        self.inaxes = None
        self.viewer = viewer
        self.figure = self.viewer.fig
        self.connect()

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if self.figure.canvas.manager.toolbar._active is None:
            if event.inaxes in self.viewer.axs:
                self.press = event.xdata, event.ydata
                self.window_ref = self.viewer.window
                self.level_ref = self.viewer.level
                self.inaxes = event.inaxes

    def on_motion(self, event):
        if (event.inaxes is not None) and (event.inaxes == self.inaxes) and (self.press is not None):
            xpress, ypress = self.press
            xlim = event.inaxes.get_xlim()
            ylim = event.inaxes.get_ylim()
            dx = (event.xdata - xpress)/(xlim[1]-xlim[0])
            dy = (event.ydata - ypress)/(ylim[1]-ylim[0])
            self.viewer.window_slider.set_val(self.window_ref * (1.0+dx))
            self.viewer.level_slider.set_val(self.level_ref * (1.0+dy))

    def on_release(self, event):
        self.press = None
        self.inaxes = None

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)


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
    window_slider = None
    level_slider = None
    window = None
    level = None
    clim = None
    window_level_mouse = None

    
    def __init__(self, im_arr, frame_dimension = None, cmap='gray'):
        
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

        self.window = np.max(self.data) - np.min(self.data)
        self.level = np.min(self.data) + self.window/2

        self.cmap = cmap

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
        
        slider_window_ax = self.fig.add_axes([0.85, 0.15, 0.10, 0.025])
        window_min = 0
        window_max = np.max(self.data)-np.min(self.data)
        self.window = window_max
        self.window_slider = widgets.Slider(slider_window_ax, label='W: ', color='black', valmin=window_min, valmax=window_max, valinit=self.window)
        self.window_slider.on_changed(lambda value: self.set_window(value))

        slider_level_ax = self.fig.add_axes([0.85, 0.10, 0.10, 0.025])
        level_max = np.max(self.data)
        level_min = np.min(self.data)
        self.level = (level_max-level_min)/2
        self.level_slider = widgets.Slider(slider_level_ax, label='L: ', color='black', valmin=level_min, valmax=level_max, valinit=self.level)
        self.level_slider.on_changed(lambda value: self.set_level(value))

        slider_f_ax = self.fig.add_axes([0.85, 0.05, 0.10, 0.025])
        self.frame_slider = widgets.Slider(slider_f_ax, label='T: ', color='black', valmin=0, valmax=self.data.shape[0]-1,valfmt=' %d',valinit=0)
        self.frame_slider.on_changed(lambda value: self.update_plot(frame=int(value)))

        self.update_contrast()
        
        self.window_level_mouse = WindowLevelMouse(self)
        
        manager = plt.get_current_fig_manager()


    def set_window(self, val):
        self.window = val
        self.update_contrast()

    def set_level(self, val):
        self.level = val
        self.update_contrast()
        

    def update_contrast(self):
        self.clim = (self.level-self.window/2, self.level+self.window/2)
        for im in self.ims:
            im.set_clim(self.clim)

    def update_plot(self, frame=0):
        for r in range(0,self.rows):
            for c in range(0,self.cols):
                im_no = r*self.cols+c
                if im_no < self.im_per_frame:
                    self.ims[im_no].set_data(np.squeeze(self.data[frame,im_no,:,:]))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ISMRMRD Image Viewer", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--time_dimension', type=int, default=-1,help='Dimension to be interpreted as time (frame)')
    parser.add_argument('-c', '--colormap', default='gray', help='Colormap')
    parser.add_argument('ismrmrd_file', help="ISMRMRD (HDF5) file")
    parser.add_argument('ismrmrd_group', help="Image group within HDF5 file")
    args = parser.parse_args()


    h, img_array = read_ismrmrd_image_series(args.ismrmrd_file, args.ismrmrd_group)

    if args.time_dimension > -1:
        v = ImageViewer(np.squeeze(img_array), args.time_dimension,cmap=args.colormap)
    else:
        v = ImageViewer(np.squeeze(img_array),cmap=args.colormap)

    plt.show()
    print("Returned from show")

if __name__ == "__main__":
    main()
