"""
Driver for finding minimum energy structures for various FvK numbers.
"""
from multiprocessing import Pool
import numpy as np
import scipy.optimize as op
from ops.core.fileio import prepareData  # , writeToVTK
from ops.core.main import energyAndJacobian


def main(dof, conn, gamma, index):
    """ Loop over asphericity and run"""
    print('Iteration ' + str(index))
    sol = op.minimize(energyAndJacobian, dof, args=(conn, gamma), jac=True,
                      method='L-BFGS-B')
    if sol.success:
        # print('\tIteration ' + index + str(sol.message))
        dof = sol.dof
        # fileName = 'Solved-{0}.vtk'.format(index)
        # writeToVTK(dof, fileName)
    else:
        pass


if __name__ == "__main__":
    CONN, X = prepareData('ops/data/T7.vtk')
    FVK = np.logspace(-2, 4, 1000)
    ARGS = [(X, CONN, gamma, index) for index, gamma in enumerate(FVK)]
    with Pool(4) as p:
        p.starmap(main, ARGS)
