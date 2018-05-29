#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from ops.core.main import energyAndJacobian, energy
from ops.core.fileio import prepareData

# Test the compute function
conn, x = prepareData('../data/T7.vtk')
Jn = np.zeros_like(x)
_,Ja = energyAndJacobian(x,conn)
hv = np.logspace(-12,-3,20)
for h in hv:
    for i in range(len(x)):
        x[i] += h
        Epv = energy(x,conn)
        Ep = sum(Epv)
        x[i] -= 2*h
        Emv = energy(x,conn)
        Em = sum(Emv)
        Jn[i] = 0.5*(Ep - Em)/h
        x[i] += h
    err = np.linalg.norm(Jn - Ja)
    print(str(err))
