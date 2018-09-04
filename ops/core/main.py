from math import sqrt, sin, cos, exp
import numpy as np
from numba import njit
from ops.core.rotations import get_orientations


@njit(cache=True)
def diffNRV(rv, Mij):
    """ Differentiate orientations wrt rotation vectors. """
    N = rv.shape[0]
    Aaj = np.zeros((4, 3))
    Bia = np.zeros((3, 4))
    eye = np.array([[0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    for i in range(N):
        r0 = rv[i, 0]
        r1 = rv[i, 1]
        r2 = rv[i, 2]

        u = sqrt(r0 * r0 + r1 * r1 + r2 * r2)
        a = 0.5 * u
        ca = cos(a)
        sau = sin(a) / u

        q0 = ca
        q1 = sau * r0
        q2 = sau * r1
        q3 = sau * r2

        Aaj[0, 0] = 2 * q2
        Aaj[0, 1] = -2 * q1
        Aaj[0, 2] = 2 * q0
        Aaj[1, 0] = 2 * q3
        Aaj[1, 1] = -2 * q0
        Aaj[1, 2] = -2 * q1
        Aaj[2, 0] = 2 * q0
        Aaj[2, 1] = 2 * q3
        Aaj[2, 2] = -2 * q2
        Aaj[3, 0] = 2 * q1
        Aaj[3, 1] = 2 * q2
        Aaj[3, 2] = 2 * q3

        Bia[0, 0] = -0.5 * sau * r0
        Bia[1, 0] = -0.5 * sau * r1
        Bia[2, 0] = -0.5 * sau * r2

        fac = (0.5 * ca - sau) / u**2
        for l in range(3):
            for m in range(1, 4):
                Bia[l, m] = sau * eye[l, m] + fac * rv[i, l] * rv[i, m-1]

        for l in range(3):
            for n in range(3):
                Mij[i, l, n] = 0.0
                for m in range(4):
                    Mij[i, l, n] += Bia[l, m] * Aaj[m, n]


@njit(cache=True)
def energyAndJacobian(x, conn, fvk=0.01, re=1.0, a=4.620981204):
    """ Calculate energy and jacobian of the energy of the OPS shell. """
    # Prepare data arrays
    energy = 0.0
    NP = int(len(x) / 6)  # Number of particles
    X = x.reshape((-1, 3))
    jac_arr = np.zeros(NP*6)
    jac = jac_arr.reshape((-1, 3))
    jacX = jac[:NP, :]
    jacR = jac[NP:, :]

    # Get the particle positions, rotation vectors and derivatives
    pos = X[:NP, :]
    rot = X[NP:, :]
    ori = np.zeros_like(rot)
    get_orientations(rot, ori)
    dNdr = np.zeros((NP, 3, 3))
    diffNRV(rot, dNdr)

    rij = np.zeros(3)
    rn = np.zeros(3)
    m = np.zeros(3)
    n = np.zeros(3)
    dMdr = np.zeros(3)
    dCdr = np.zeros(3)
    dPhi_cVi = np.zeros(3)
    dPhi_cVj = np.zeros(3)
    dPhi_nVi = np.zeros(3)
    dPhi_nVj = np.zeros(3)

    # Iterate over all edges from connectivity and calculate
    for z in range(conn.shape[0]):

        # Get the point ids
        i = conn[z, 0]
        j = conn[z, 1]

        # Get the positions and orientations and derivatives of normals
        for k in range(3):
            rij[k] = pos[j, k] - pos[i, k]
            m[k] = ori[i, k] - ori[j, k]
            n[k] = ori[i, k] + ori[j, k]

        r = sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2])
        inv_r = 1.0 / r

        n_dot_rn = 0.0
        m_dot_m = 0.0
        for k in range(3):
            rn[k] = rij[k] * inv_r
            n_dot_rn += n[k] * rn[k]
            m_dot_m += m[k] * m[k]

        # Morse potential and Co-circularity
        exp_1 = exp(-a*(r - re))
        exp_2 = exp_1**2

        # Morse and Co-circularity derivatives
        fac = 2*a*(exp_1 - exp_2)
        fac2 = 2 * n_dot_rn * inv_r
        for k in range(3):
            dMdr[k] = fac * rn[k]
            dCdr[k] = fac2 * (n[k] - n_dot_rn * rn[k])

        # Matrix vector scalar multiplications
        #  Co-normality derivatives
        #  Co-circularity potential derivatives
        fac = 2 * n_dot_rn
        for p in range(3):
            dPhi_nVi[p] = 0.0
            dPhi_nVj[p] = 0.0
            dPhi_cVi[p] = 0.0
            dPhi_cVj[p] = 0.0
            for q in range(3):
                dPhi_nVi[p] += 2.0 * dNdr[i, p, q] * m[q]
                dPhi_nVj[p] -= 2.0 * dNdr[j, p, q] * m[q]
                dPhi_cVi[p] += fac * dNdr[i, p, q] * rn[q]
                dPhi_cVj[p] += fac * dNdr[j, p, q] * rn[q]

        # Total energy
        inv_fvk = 1.0 / fvk
        energy += (exp_2 - 2*exp_1) + (m_dot_m + n_dot_rn**2) * inv_fvk

        # Total Derivatives of energy wrt xi, vi, vj
        for k in range(3):
            jacX[i, k] -= dMdr[k] + dCdr[k] * inv_fvk
            jacX[j, k] += dMdr[k] + dCdr[k] * inv_fvk
            jacR[i, k] += (dPhi_nVi[k] + dPhi_cVi[k]) * inv_fvk
            jacR[j, k] += (dPhi_nVj[k] + dPhi_cVj[k]) * inv_fvk

    return energy, jac_arr


@njit(cache=True)
def energy(x, conn, fvk=0.01, re=1.0, a=4.620981204):
    """ Calculate the more, normality and circularity energy. """
    # Prepare data arrays
    NP = int(len(x) / 6)  # Number of particles
    morse = 0
    normality = 0
    circularity = 0
    X = x.reshape((-1, 3))

    # Get the particle positions, rotation vectors and derivatives
    pos = X[:NP, :]
    rot = X[NP:, :]
    ori = np.zeros_like(rot)
    get_orientations(rot, ori)

    # Iterate over all edges from connectivity and calculate
    for z in range(conn.shape[0]):

        # Get the point ids
        i, j = conn[z]

        # Get the positions and orientations and derivatives of normals
        rij = pos[j] - pos[i]
        r = np.linalg.norm(rij)
        rn = rij / r

        # Morse potential and derivatives
        exp_1 = np.exp(-a*(r - re))
        morse += exp_1*(exp_1 - 2)

        # Co-normality potential and derivatives
        m = ori[i] - ori[j]
        normality += np.dot(m, m)/fvk

        # Co-circularity potential and derivatives
        n_dot_rn = np.dot(ori[i] + ori[j], rn)
        circularity += (n_dot_rn**2) / fvk

    return morse, normality, circularity
