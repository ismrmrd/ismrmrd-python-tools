"""
Simple tiled image display
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def imshow(image_matrix, tile_shape=None, scale=None, titles=[],
           colorbar=False, cmap='jet'):
    """ Tile images and display them in a window.

    Paramters
    ---------
    image_matrix : array
        a 2D or 3D set of image data
    tile_shape : array or None, optional
        optional shape `(rows, cols)` for tiling images
    scale : tuple or None, optional
        optional `(min, max)` values for scaling all images
    titles : list or None, optional
        optional list of titles for each subplot
    cmap : str or `matplotlib.colors.Colormap`, optional
        optional colormap for all images
    """
    if image_matrix.ndim not in [2, 3]:
        raise ValueError("image_matrix must have 2 or 3 dimensions")

    if image_matrix.ndim == 2:
        image_matrix = image_matrix.reshape(
            (1, image_matrix.shape[0], image_matrix.shape[1]))

    if not scale:
        scale = (np.min(image_matrix), np.max(image_matrix))
    vmin, vmax = scale

    if not tile_shape:
        tile_shape = (1, image_matrix.shape[0])
    if np.prod(tile_shape) < image_matrix.shape[0]:
        raise ValueError("image tile rows x columns must equal the 3rd dim "
                         "extent of image_matrix")

    # add empty titles as necessary
    if len(titles) < image_matrix.shape[0]:
        titles.extend(['' for x in range(image_matrix.shape[0] - len(titles))])

    if (len(titles) > 0) and (len(titles) < image_matrix.shape[0]):
        raise ValueError("number of titles must equal 3rd dim extent of "
                         "image_matrix")

    def onselect(eclick, erelease):
        print((eclick.xdata, eclick.ydata), (erelease.xdata, erelease.ydata))

    def on_pick(event):
        if isinstance(event.artist, matplotlib.image.AxesImage):
            x, y = event.mouseevent.xdata, event.mouseevent.ydata
            im = event.artist
            A = im.get_array()
            print(A[y, x])

    selectors = []  # need to keep a reference to each selector
    rectprops = dict(facecolor='red', edgecolor='black', alpha=0.5, fill=True)
    cols, rows = tile_shape
    fig = plt.figure()
    plt.set_cmap(cmap)
    for z in range(image_matrix.shape[0]):
        ax = fig.add_subplot(cols, rows, z+1)
        ax.set_title(titles[z])
        ax.set_axis_off()
        imgplot = ax.imshow(
            image_matrix[z, :, :], vmin=vmin, vmax=vmax, picker=True)
        selectors.append(RectangleSelector(ax, onselect, rectprops=rectprops))

        if colorbar is True:
            plt.colorbar(imgplot)

    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.show()
