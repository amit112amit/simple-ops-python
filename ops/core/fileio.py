import numpy as np
import vtk as v
import vtk.numpy_interface.dataset_adapter as dsa
import vtk.numpy_interface.algorithms as alg
from ops.core.rotations import get_orientations, get_rotation_vectors
from ops.core.meshmethods import triangulate, getEdges, averageEdgeLength

def prepareData(fileName):
    """ Read a VTK file and create orientations and rotation vectors. """
    reader = v.vtkPolyDataReader()
    reader.SetFileName( fileName )
    reader.Update()
    polyData = dsa.WrapDataObject(reader.GetOutput())

    # Triangulate
    polyData.SetPolys( triangulate(polyData) )

    # Prepare positions, orientations and rotation vectors
    positions = polyData.Points
    orientations = alg.norm( positions )
    rotationVectors = np.zeros_like( orientations )
    get_rotation_vectors( orientations, rotationVectors )

    # Get the connectivity matrix for the edges
    conn = getEdges( polyData )

    # Find the average edge length and normalize by it
    avg = averageEdgeLength( positions, conn )
    positions = positions/avg

    x = np.concatenate( ( positions, rotationVectors ) ).ravel()
    return conn, x

def writeToVTK(x,fileName):
    """ Write flattened vector to VTK file. """
    X = x.reshape((-1,3))
    pos = X[:72,:]
    posArr = dsa.numpyTovtkDataArray(pos,name='Points')
    rot = X[72:,:]
    ori = np.zeros_like(rot)
    get_orientations( rot, ori )
    normals = dsa.numpyTovtkDataArray(ori,name='PointNormals')
    pts = v.vtkPoints()
    pts.SetData( posArr )
    pd = v.vtkPolyData()
    pd.SetPoints( pts )
    pd.GetPointData().SetNormals(normals)
    polyData = dsa.WrapDataObject( pd )
    polyData.SetPolys( triangulate(polyData) )
    w = v.vtkPolyDataWriter()
    w.SetFileName(fileName)
    w.SetInputData( pd )
    w.Write()
