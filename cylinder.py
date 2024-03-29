#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from bumps.names import Parameter
from sans.dataloader.loader import Loader
from sans.dataloader.manipulations import Ringcut
from ccode import GpuCylinder, CylinderParameters

def load_data(filename):
    loader = Loader()
    data = loader.load(filename)
    return data

def set_beam_stop(data, radius):
    data.mask = Ringcut(0, radius)(data)

def plot_data(data, iq):
    from numpy.ma import masked_array
    import matplotlib.pyplot as plt
    img = masked_array(iq, data.mask)
    xmin, xmax = min(data.qx_data), max(data.qx_data)
    ymin, ymax = min(data.qy_data), max(data.qy_data)
    plt.imshow(img.reshape(128,128),
               interpolation='nearest', aspect=1, origin='upper',
               extent=[xmin, xmax, ymin, ymax])
def plot_result(data, theory):
    import matplotlib.pyplot as plt
    plt.subplot(1,3,1)
    plot_data(data, data.data)
    plt.subplot(1,3,2)
    plot_data(data, theory)
    plt.subplot(1,3,3)
    plot_data(data, (theory-data.data)/data.err_data)

def demo():
    data = load_data('JUN03289.DAT')
    set_beam_stop(data, 0.004)
    plot_data(data)
    import matplotlib.pyplot as plt; plt.show()

PARS = {
    'scale':1,'radius':1,'length':1,'sldCyl':1e-6,'sldSolv':0,'background':0,
    'cyl_theta':0,'cyl_phi':0,'M0_sld_cyl':1e-33,'M0_sld_solv':1e-33,
}
class Ellipse(object):
    def __init__(self, data, **kw):
        self.index = data.mask==0
        self.iq = data.data[self.index]
        self.diq = data.err_data[self.index]
        self.data = data
        self.qx = data.qx_data
        self.qy = data.qy_data
        self.gpu = GpuCylinder(self.qx, self.qy)
        extra_pars = set(kw.keys()) - set(PARS.keys())
        if extra_pars:
            raise TypeError("unexpected parameters %s"%(str(extra_pars,)))
        pars = PARS.copy()
        pars.update(kw)
        self._parameters = dict((k,Parameter(v,name=k)) for k,v in pars.items())

    def numpoints(self):
        return len(self.iq)

    def parameters(self):
        return self._parameters

    def __getattr__(self, par):
        return self._parameters[par]

    def theory(self):
        #return self.parts[0](self.X,self.Y)
        #parts = [M(self.X,self.Y) for M in self.parts]
        #for i,p in enumerate(parts):
        #    if np.any(np.isnan(p)): print "NaN in part",i
        kw = dict((k,v.value) for k,v in self._parameters.items())
        pars = CylinderParameters(**kw)
        result = self.gpu.cylinder_fit(self.qx, self.qy, pars, r_n=35, t_n=35, l_n=1, p_n=1, b_w=.1, t_w=.1, a_w=.1, p_w=.1, sigma=3)
        return result

    def residuals(self):
        #if np.any(self.err ==0): print "zeros in err"
        return (self.theory()[self.index]-self.iq)/self.diq

    def nllf(self):
        R = self.residuals()
        #if np.any(np.isnan(R)): print "NaN in residuals"
        return 0.5*np.sum(R**2)

    def __call__(self):
        return 2*self.nllf()/self.dof

    def plot(self, view='linear'):
        plot_result(self.data, self.theory())

    def save(self, basename):
        pass

    def update(self):
        pass
