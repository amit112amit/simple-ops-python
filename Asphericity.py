import numpy as np
import scipy.optimize as op
from ops import prepareData, compute, writeToVTK

conn, x = prepareData('T7.vtk')

fvk = np.logspace(-2,4,1000)

for i,g in enumerate(fvk):
    print('Iteration' + str(i))
    sol = op.minimize( compute, x, args=(conn,g), jac=True, method='L-BFGS-B' )
    if sol.success:
        print('\t' + str(sol.message))
        x = sol.x
        fileName = 'Solved-{0}.vtk'.format(i)
        writeToVTK(x,fileName)
    else:
        print('Solution failed' + str(sol.message))