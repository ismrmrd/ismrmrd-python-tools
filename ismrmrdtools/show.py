import numpy as np
import matplotlib.pyplot as plt

def imshow(image_matrix, tile_shape=None, scale=None, titles=[]):
    """ Tiles images and displays them in a window.

    :param image_matrix: a 2D or 3D set of image data
    :param tile_shape: optional shape (rows, cols) for tiling images
    :param scale: optional (min,max) values for scaling all images
    :param titles: optional list of titles for each subplot
    """
    assert image_matrix.ndim in [2, 3], "image_matrix must have 2 or 3 dimensions"

    if image_matrix.ndim == 2:
        image_matrix = image_matrix.reshape((1, image_matrix.shape[0], image_matrix.shape[1]))

    if not scale:
        scale = (np.min(image_matrix), np.max(image_matrix))
    vmin, vmax = scale

    if not tile_shape:
        tile_shape = (1, image_matrix.shape[0])
    assert np.prod(tile_shape) >= image_matrix.shape[0],\
            "image tile rows x columns must equal the 3rd dim extent of image_matrix"

    # add empty titles as necessary
    if len(titles) < image_matrix.shape[0]:
        titles.extend(['' for x in range(image_matrix.shape[0] - len(titles))])

    if len(titles) > 0:
        assert len(titles) >= image_matrix.shape[0],\
                "number of titles must equal 3rd dim extent of image_matrix"

    cols, rows = tile_shape
    print(cols, rows)
    fig = plt.figure()
    for z in range(image_matrix.shape[0]):
        a = fig.add_subplot(cols, rows, z+1)
        a.set_title("%d: %s" % (z, titles[z]))
        a.set_axis_off()
        plt.imshow(image_matrix[z,:,:], vmin=vmin, vmax=vmax)
    plt.show()
