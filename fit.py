#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bumps.names import *
import cylinder

data = cylinder.load_data('JUNO3289.DAT')
cylinder.set_beam_stop(data, 0.004)


model = cylinder.Cylinder(data, scale=1, radius=64.1, length=266.96, sldCyl=.291e-6, sldSolv=5.77e-6, background=0,
                              cyl_theta=0, cyl_phi=0, M0_sld_cyl=1.0e-33, M0_sld_solv=1.0e-33)
model.scale.range(0,10)

problem = FitProblem(model)