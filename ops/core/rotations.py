import numpy as np
from numba import jit

@jit(nopython=True,cache=True)
def get_orientations(rotVec,orientations):
    """ Convert Rotation Vectors to Orientations"""
    for i in range( rotVec.shape[0] ):
        r = rotVec[i]
        R = np.linalg.norm(r) # Angle of rotation
        # Check 0-angle case
        if R < 1e-10:
            orientations[i] = np.array( [0.0, 0.0, 1.0] )
        elif abs(R - np.pi) < 1e-10:
            orientations[i] = np.array( [0.0, 0.0, -1.0] )
        else:
            # Make unit quaternion with angle R and axis along rotVec
            q0 = np.cos( 0.5*R )
            q1, q2, q3 = np.sin( 0.5*R )*r/R
            orientations[i] = np.array([2*(q0*q2 + q1*q3), -2*(q0*q1 - q2*q3),
                                        q0**2 - q1**2 - q2**2 + q3**2])

@jit(nopython=True,cache=True)
def get_rotation_vectors(orientations,rotVec):
    """ Convert Orientations to Rotation Vectors """
    for i in range(orientations.shape[0]):
        p0, p1, p2 = orientations[i]
        # Cross-product of ptNormal with z-axis.
        cross_prod = np.array( [-p1, p0, 0.0] )
        cross_prod_norm = np.sqrt( p1**2 + p0**2 )
        # Check if the orientation is parallel or anti-parallel to z-axis
        if cross_prod_norm < 1e-10:
            axis = np.array([ 1.0, 0.0, 0.0 ]) #choose the x-axis
            angle = 0.0 if p2 > 0.0 else np.pi
        else:
            angle = np.arcsin( cross_prod_norm )
            angle = (np.pi - angle) if p2 < 0.0 else angle
            axis = cross_prod/cross_prod_norm
        rotVec[i] = angle*axis
