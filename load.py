#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sans.dataloader.loader import Loader
from sans.dataloader.manipulations import Ringcut

def load_data(filename):
    loader = Loader()
    f = loader.load(filename)
    return f

def set_beam_stop(data, radius):
    data.mask = Ringcut(0, radius)(data)

def plot_data(data):
    from numpy.ma import masked_array
    import matplotlib.pyplot as plt
    img = masked_array(data.data, data.mask)
    xmin, xmax = min(data.qx_data), max(data.qx_data)
    ymin, ymax = min(data.qy_data), max(data.qy_data)
    plt.imshow(img.reshape(128,128),
               interpolation='nearest', aspect=1, origin='upper',
               extent=[xmin, xmax, ymin, ymax])

def demo():
    data = load_data('JUN03289.DAT')
    set_beam_stop(data, 0.004)
    plot_data(data)
    import matplotlib.pyplot as plt; plt.show()


if __name__ == "__main__":
    demo()

