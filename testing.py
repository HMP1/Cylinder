#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import pyopencl as cl
from time import time
import matplotlib.pyplot as plt

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

def cylinder_fit(pars, r_n, t_n, l_n, p_n, r_w=.1, t_w=.1, l_w=.1, p_w=.1, sigma=3, qx_size=128):
    radius = GaussianDispersion(r_n, r_w, sigma)
    length = GaussianDispersion(l_n, l_w, sigma) #this is an L
    cyl_theta = GaussianDispersion(t_n, t_w, sigma)
    cyl_phi = GaussianDispersion(p_n, p_w, sigma)
    #Get the weights for each
    radius.value, radius.weight = radius.get_weights(pars.radius, 0, 1000, True)
    length.value, length.weight = length.get_weights(pars.length, 0, 1000, True)
    cyl_theta.value, cyl_theta.weight = cyl_theta.get_weights(pars.cyl_theta, 0, 90, False)
    cyl_phi.value, cyl_phi.weight = cyl_phi.get_weights(pars.cyl_phi, 0, 90, False)

    #create qx and qy evenly spaces
    qx = np.linspace(-.02, .02, qx_size)
    qy = np.linspace(-.02, .02, qx_size)
    qx, qy = np.meshgrid(qx, qy)

    #saved shape of qx
    r_shape = qx.shape

    #reshape for calculation; resize as float32
    qx = np.reshape(qx, [qx.size])
    qy = np.reshape(qy, [qy.size])
    qx = np.asarray(qx, np.float32)
    qy = np.asarray(qy, np.float32)
    total = qx.size

    #create context, queue, and build program
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    src = open('Kernel-Cylinder.cpp').read()
    prg = cl.Program(ctx, src).build()

    #buffers
    mf = cl.mem_flags
    qx_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=qx)
    qy_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=qy)
    res_b = cl.Buffer(ctx, mf.WRITE_ONLY, qx.nbytes)
    res = np.empty_like(qx)

    #Perform the computation, with all weight points
    sum, norm, norm_vol, vol = 0.0, 0.0, 0.0, 0.0
    size = len(cyl_theta.weight)

    #Loop over radius, length, theta, phi weight points
    for i in xrange(len(radius.weight)):

        for j in xrange(len(length.weight)):

            for k in xrange(len(cyl_theta.weight)):

                for l in xrange(len(cyl_phi.weight)):

                    prg.CylinderKernel(queue, qx.shape, None, qx_b, qy_b, res_b, np.float32(pars.sldSolv),
                                       np.float32(pars.sldCyl), np.float32(pars.M0_sld_cyl), np.float32(pars.M0_sld_cyl),
                                       np.float32(radius.value[i]), np.float32(length.value[j]), np.float32(pars.scale),
                                       np.float32(radius.weight[i]), np.float32(length.weight[j]), np.float32(cyl_theta.weight[k]),
                                       np.float32(cyl_phi.weight[l]), np.float32(cyl_theta.value[k]), np.float32(cyl_phi.value[l]),
                                       np.float32(pars.background), np.uint32(total), np.uint32(size))
                    cl.enqueue_copy(queue, res, res_b)

    # Averaging in theta needs an extra normalization
    # factor to account for the sin(theta) term in the
    # integration (see documentation).

    return res


#int main()
pars = CylinderParameters(scale=1, radius=64.1, length=266.96, sldCyl=.291e-6, sldSolv=5.77e-6, background=0,
                          cyl_theta=0, cyl_phi=90, M0_sld_cyl=1.0e-33, M0_sld_solv=1.0e-33)
t = time()
result = cylinder_fit(pars, r_n=2, t_n=2, l_n=2, p_n=2, sigma=3, r_w=.1, t_w=.1, l_w=.1, p_w=.1, qx_size=128)
tt = time()

print("Time taken: %f" % (tt - t))



a = open("cylinder.txt", "w")
for x in xrange(len(result)):
    a.write(str(result))
    a.write("\n")





























