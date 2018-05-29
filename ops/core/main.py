import numpy as np
from numba import jit
from ops.core.meshmethods import triangulate, averageEdgeLength, getEdges
from ops.core.rotations import get_orientations

@jit(nopython=True,cache=True)
def diffNRV( rv, Mij ):
    """ Differentiate orientations wrt rotation vectors. """
    N = rv.shape[0]
    for i in range(N):
        u = np.linalg.norm( rv[i] )
        a = 0.5*u
        ca = np.cos(a)
        sau = np.sin(a)/u
        q0 = ca
        q1, q2, q3 = sau*rv[i]
        Aaj = 2*np.array([q2,-q1,q0,q3,-q0,-q1,q0,q3,-q2,q1,q2,q3]).reshape(
            (4,3))
        Bia = np.zeros( (3,4) )
        Bia[:,0] = -0.5*sau*rv[i]
        Bia[:,1:] = sau*np.eye(3) + (0.5*ca - sau)*np.outer(rv[i],rv[i])/u**2
        Mij[:,:,i] = np.dot( Bia, Aaj)

@jit(nopython=True,cache=True)
def energyAndJacobian( x, conn, fvk=0.01, re=1.0, a=4.620981204 ):
    """ Calculate energy and jacobian of the energy of the OPS shell. """
    # Prepare data arrays
    NP = int( len(x) / 6 ) # Number of particles
    energy = 0
    X = x.reshape( (-1,3) )
    jac = np.zeros_like( X )
    jacX = jac[:NP,:]
    jacR = jac[NP:,:]

    # Get the particle positions, rotation vectors and derivatives
    pos = X[:NP,:]
    rot = X[NP:,:]
    ori = np.zeros_like( rot )
    get_orientations( rot, ori )
    dNdr = np.zeros( (3,3,NP) )
    diffNRV( rot, dNdr )

    # Iterate over all edges from connectivity and calculate
    for z in range(conn.shape[0]):
        # Get the point ids
        i, j = conn[z]
        # Get the positions and orientations and derivatives of normals
        rij =  pos[j] - pos[i]
        r = np.linalg.norm(rij)
        rn = rij/r
        m = ori[i] - ori[j]
        n = ori[i] + ori[j]
        M = dNdr[:,:,i]
        N = dNdr[:,:,j]
        n_dot_rn = np.dot(n,rn)
        # Morse potential and derivatives
        exp_1 = np.exp( -a*(r - re) )
        exp_2 = exp_1**2
        dMdr = 2*a*(exp_1 - exp_2)*rn
        # Co-normality potential and derivatives
        dPhi_nVi = 2*np.dot(M,m)
        dPhi_nVj = -2*np.dot(N,m)
        # Co-circularity potential and derivatives
        dCdr = 2*n_dot_rn*(n - n_dot_rn*rn)/r
        dPhi_cVi = 2*n_dot_rn*np.dot(M,rn)
        dPhi_cVj = 2*n_dot_rn*np.dot(N,rn)
        # Total energy
        energy += (exp_2 - 2*exp_1) + ( np.dot(m,m) + n_dot_rn**2 )/fvk
        # Total Derivatives of energy wrt xi, vi, vj
        Dxi = -(dMdr + dCdr/fvk)
        Dvi = (dPhi_nVi + dPhi_cVi)/fvk
        Dvj = (dPhi_nVj + dPhi_cVj)/fvk
        # Update the jacobian
        jacX[i] += Dxi
        jacX[j] += -Dxi
        jacR[i] += Dvi
        jacR[j] += Dvj

    return energy, jac.ravel()


@jit(nopython=True,cache=True)
def energy( x, conn, fvk=0.01, re=1.0, a=4.620981204 ):
    """ Calculate the more, normality and circularity energy. """
    # Prepare data arrays
    NP = int( len(x) / 6 ) # Number of particles
    morse = 0
    normality = 0
    circularity = 0
    X = x.reshape( (-1,3) )

    # Get the particle positions, rotation vectors and derivatives
    pos = X[:NP,:]
    rot = X[NP:,:]
    ori = np.zeros_like( rot )
    get_orientations( rot, ori )

    # Iterate over all edges from connectivity and calculate
    for z in range(conn.shape[0]):

        # Get the point ids
        i, j = conn[z]

        # Get the positions and orientations and derivatives of normals
        rij =  pos[j] - pos[i]
        r = np.linalg.norm(rij)
        rn = rij/r

        # Morse potential and derivatives
        exp_1 = np.exp( -a*(r - re) )
        morse += exp_1*(exp_1 - 2)

        # Co-normality potential and derivatives
        m = ori[i] - ori[j]
        normality += np.dot(m,m)/fvk

        # Co-circularity potential and derivatives
        n_dot_rn = np.dot( ori[i] + ori[j], rn )
        circularity +=  ( n_dot_rn**2 )/fvk

    return morse, normality, circularity
