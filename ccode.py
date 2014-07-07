#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import pyopencl as cl

class CylinderParameters:
    def __init__(self, scale, radius, length, sldCyl, sldSolv, background, cyl_theta, cyl_phi, M0_sld_cyl,  M0_sld_solv): #Up_theta,  M_phi_cyl, M_theta_cyl, M_phi_solv, M_theta_solv, Up_frac_i, Up_frac_f):
        self.scale = scale
        self.radius = radius
        self.length = length
        self.sldCyl = sldCyl
        self.sldSolv = sldSolv
        self.background = background
        self.cyl_theta = cyl_theta
        self.cyl_phi = cyl_phi
        self.M0_sld_cyl = M0_sld_cyl
        self.M0_sld_solv = M0_sld_solv
"""     self.Up_theta = Up_theta
        self.M_phi_cyl = M_phi_cyl
        self.M_theta_cyl = M_theta_cyl
        self.M_phi_solv = M_phi_solv
        self.M_theta_solv = M_theta_solv
        self.Up_frac_i = Up_frac_i
        self.Up_frac_f = Up_frac_f """

class GaussianDispersion(object):
    def __init__(self, npts=35, width=0, nsigmas=3): #number want, percent deviation, #standard deviations from mean
        self.type = 'gaussian'
        self.npts = npts
        self.width = width
        self.nsigmas = nsigmas

    def get_pars(self):
        return self.__dict__

    def get_weights(self, center, min, max, relative):
        """ *center* is the center of the distribution
        *min*,*max* are the min, max allowed values
        *relative* is True if the width is relative to the center instead of absolute
        For polydispersity use relative.  For orientation parameters use absolute."""
        npts, width, nsigmas = self.npts, self.width, self.nsigmas
        sigma = width * center if relative else width
        if sigma == 0:
            return np.array([center, 1.], 'd')
        x = center + np.linspace(-nsigmas * sigma, +nsigmas * sigma, npts)
        x = x[(x >= min) & (x <= max)]
        px = np.exp((x-center)**2 / (-2.0 * sigma * sigma))
        return x, px

class GpuCylinder(object):
    def __init__(self, qx, qy):

        self.qx = np.asarray(qx, np.float32)
        self.qy = np.asarray(qy, np.float32)
        #create context, queue, and build program
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, open('Kernel-Cylinder.cpp').read()).build()

        #buffers
        mf = cl.mem_flags
        self.qx_b = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.qx)
        self.qy_b = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.qy)
        self.res_b = cl.Buffer(self.ctx, mf.WRITE_ONLY, qx.nbytes)
        self.res = np.empty_like(self.qx)

    def cylinder_fit(self, qx, qy, pars, r_n, t_n, l_n, p_n, r_w=.1, t_w=.1, l_w=.1, p_w=.1, sigma=3):

        radius = GaussianDispersion(r_n, r_w, sigma)
        length = GaussianDispersion(l_n, l_w, sigma) #this is an L
        cyl_theta = GaussianDispersion(t_n, t_w, sigma)
        cyl_phi = GaussianDispersion(p_n, p_w, sigma)
        #Get the weights for each
        radius.value, radius.weight = radius.get_weights(pars.radius, 0, 1000, True)
        length.value, length.weight = length.get_weights(pars.length, 0, 1000, True)
        cyl_theta.value, cyl_theta.weight = cyl_theta.get_weights(pars.cyl_theta, 0, 90, False)
        cyl_phi.value, cyl_phi.weight = cyl_phi.get_weights(pars.cyl_phi, 0, 90, False)

        #Perform the computation, with all weight points
        sum, norm, norm_vol, vol = 0.0, 0.0, 0.0, 0.0
        size = len(cyl_theta.weight)

        #Loop over radius, length, theta, phi weight points
        for i in xrange(len(radius.weight)):
            for j in xrange(len(length.weight)):
                for k in xrange(len(cyl_theta.weight)):
                    for l in xrange(len(cyl_phi.weight)):

                        self.prg.CylinderKernel(self.queue, qx.shape, None, self.qx_b, self.qy_b, self.res_b, np.float32(pars.sldSolv),
                                           np.float32(pars.sldCyl), np.float32(pars.M0_sld_cyl), np.float32(pars.M0_sld_cyl),
                                           np.float32(radius.value[i]), np.float32(length.value[j]), np.float32(pars.scale),
                                           np.float32(radius.weight[i]), np.float32(length.weight[j]), np.float32(cyl_theta.weight[k]),
                                           np.float32(cyl_phi.weight[l]), np.float32(cyl_theta.value[k]), np.float32(cyl_phi.value[l]),
                                           np.float32(pars.background), np.uint32(qx.size), np.uint32(size))
                        cl.enqueue_copy(self.queue, self.res, self.res_b)
                        sum += self.res
                        vol += radius.weight[i]*length.weight[j]*pow(radius.value[i], 2)*length.value[j]
                        norm_vol += radius.weight[i]*length.weight[j]
                        norm += radius.weight[i]*length.weight[j]*cyl_theta.weight[k]*cyl_phi.weight[l]

        # Averaging in theta needs an extra normalization
        # factor to account for the sin(theta) term in the
        # integration (see documentation).

        if size > 1:
            norm /= math.asin(1.0)
        if vol != 0.0 and norm_vol != 0.0:
            sum *= norm_vol/vol

        final = sum/norm+pars.background

        return final

def demo():
    from time import time
    import matplotlib.pyplot as plt

    #create qx and qy evenly spaces
    qx = np.linspace(-.02, .02, 128)
    qy = np.linspace(-.02, .02, 128)
    qx, qy = np.meshgrid(qx, qy)

    #saved shape of qx
    r_shape = qx.shape

    #reshape for calculation; resize as float32
    qx = qx.flatten()
    qy = qy.flatten()

    pars = CylinderParameters(scale=1, radius=64.1, length=266.96, sldCyl=.291e-6, sldSolv=5.77e-6, background=0,
                              cyl_theta=0, cyl_phi=0, M0_sld_cyl=1.0e-33, M0_sld_solv=1.0e-33)
    t = time()
    result = GpuCylinder(qx, qy)
    result.x = result.cylinder_fit(qx, qy, pars, r_n=10, t_n=10, l_n=10, p_n=10, sigma=3, r_w=.1, t_w=.1, l_w=.1, p_w=.1)
    result.x = np.reshape(result.x, r_shape)
    tt = time()

    print("Time taken: %f" % (tt - t))

    plt.pcolormesh(result.x)
    plt.show()

if __name__=="__main__":
    demo()



























