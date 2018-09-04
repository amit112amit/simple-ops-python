from math import sin, cos, sqrt, asin, pi
import numpy as np
from numba import njit


@njit(cache=True)
def get_orientations(rotvec, orientations):
    """ Convert Rotation Vectors to Orientations"""
    for i in range(rotvec.shape[0]):
        r0 = rotvec[i, 0]
        r1 = rotvec[i, 1]
        r2 = rotvec[i, 2]

        R = sqrt(r0**2 + r1**2 + r2**2)
        # Check 0-angle case
        if R < 1e-10:
            orientations[i, 0] = 0.0
            orientations[i, 1] = 0.0
            orientations[i, 2] = 1.0
        elif abs(R - np.pi) < 1e-10:
            orientations[i, 0] = 0.0
            orientations[i, 1] = 0.0
            orientations[i, 2] = -1.0
        else:
            # Make unit quaternion with angle R and axis along rotvec
            q0 = cos(0.5 * R)
            sinR = sin(0.5 * R) / R
            q1 = sinR * r0
            q2 = sinR * r1
            q3 = sinR * r2
            orientations[i, 0] = 2*(q0*q2 + q1*q3)
            orientations[i, 1] = -2*(q0*q1 - q2*q3)
            orientations[i, 2] = q0**2 - q1**2 - q2**2 + q3**2


@njit(cache=True)
def get_rotation_vectors(orientations, rotvec):
    """ Convert Orientations to Rotation Vectors """
    for i in range(orientations.shape[0]):
        p0 = orientations[i, 0]
        p1 = orientations[i, 1]
        p2 = orientations[i, 2]
        # Cross-product of ptNormal with z-axis.
        cross_prod0 = -p1
        cross_prod1 = p0
        cross_prod2 = 0.0
        cross_prod_norm = sqrt(p1**2 + p0**2)
        # Check if the orientation is parallel or anti-parallel to z-axis
        if cross_prod_norm < 1e-10:
            axis0 = 1.0
            axis1 = 0.0
            axis2 = 0.0  # choose the x-axis
            angle = 0.0 if p2 > 0.0 else pi
        else:
            angle = asin(cross_prod_norm)
            angle = (pi - angle) if p2 < 0.0 else angle
            axis0 = cross_prod0/cross_prod_norm
            axis1 = cross_prod1/cross_prod_norm
            axis2 = cross_prod2/cross_prod_norm

        rotvec[i, 0] = angle * axis0
        rotvec[i, 1] = angle * axis1
        rotvec[i, 2] = angle * axis2
