import warnings
warnings.filterwarnings("ignore")
import numpy as np
import vtk as v
import vtk.numpy_interface.dataset_adapter as dsa
import vtk.numpy_interface.algorithms as alg
from numba import jit

def triangulate(pd):
    """
    Generates a triangle mesh for a spherical point cloud. It is assumed that
    'pd' is an object of dsa.PolyData type.
    """
    # Project on a sphere
    sphereXyz = alg.norm(pd.Points)*np.linalg.norm( pd.Points, axis=1 ).mean()
    sphereXyz = np.around(sphereXyz,decimals=2)
    sphereArr = dsa.numpyTovtkDataArray( sphereXyz, name='SpherePts' )
    pts = v.vtkPoints()
    pts.SetData( sphereArr )
    sphere = v.vtkPolyData()
    sphere.SetPoints( pts )

    # Store the original point ids
    idf = v.vtkIdFilter()
    idf.SetIdsArrayName('PointIds')
    idf.PointIdsOn()
    idf.SetInputData( sphere )

    # Delaunay3D to make a convex hull
    d3d = v.vtkDelaunay3D()
    d3d.SetInputConnection( idf.GetOutputPort() )

    # Extract the surface
    surf = v.vtkDataSetSurfaceFilter()
    surf.SetInputConnection( d3d.GetOutputPort() )
    surf.Update()

    # Now make a new cell array mapping to the old ids
    polyCells = v.vtkCellArray()
    sphereCells = surf.GetOutput().GetPolys()
    sphereCells.InitTraversal()
    origIds = surf.GetOutput().GetPointData().GetArray('PointIds')
    ptIds = v.vtkIdList()
    while( sphereCells.GetNextCell( ptIds ) ):
        polyCells.InsertNextCell(3)
        polyCells.InsertCellPoint( int(origIds.GetTuple1( ptIds.GetId(0) )) )
        polyCells.InsertCellPoint( int(origIds.GetTuple1( ptIds.GetId(1) )) )
        polyCells.InsertCellPoint( int(origIds.GetTuple1( ptIds.GetId(2) )) )

    return polyCells

def getEdges(pd):
    """ Extracts edges from polydata with correct point ids. """
    # Store the original point ids
    idf = v.vtkIdFilter()
    idf.PointIdsOn()
    idf.SetIdsArrayName('PointIds')
    idf.SetInputData( pd.VTKObject )

    # Extract the edges
    edgeFilter = v.vtkExtractEdges()
    edgeFilter.SetInputConnection(idf.GetOutputPort())
    edgeFilter.Update()

    # Now make a new cell array mapping to the old ids
    edges = v.vtkCellArray()
    badEdges = edgeFilter.GetOutput().GetLines()
    badEdges.InitTraversal()
    origIds = edgeFilter.GetOutput().GetPointData().GetArray('PointIds')
    ptIds = v.vtkIdList()
    while( badEdges.GetNextCell( ptIds ) ):
        edges.InsertNextCell(2)
        edges.InsertCellPoint( int(origIds.GetTuple1( ptIds.GetId(0) )) )
        edges.InsertCellPoint( int(origIds.GetTuple1( ptIds.GetId(1) )) )

    # Convert the cell array into a numpy array
    conn = dsa.vtkDataArrayToVTKArray( edges.GetData() ).reshape(
        (edges.GetNumberOfCells(),3) )[:,1:]
    return conn

@jit(nopython=True,cache=True)
def averageEdgeLength( positions, conn ):
    """ Calculate average edge length and normalize the positions by it. """
    avg = 0
    for z in range( conn.shape[0] ):
        i, j = conn[z]
        avg += np.linalg.norm( positions[i] - positions[j] )
    avg /= conn.shape[0]
    return avg

