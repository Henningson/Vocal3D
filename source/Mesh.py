
from geomdl import BSpline
from geomdl import utilities
from torch_nurbs_eval.surf_eval import SurfEval
import numpy as np
import torch
from scipy.spatial import Delaunay


def generate_surface_points(control_points, zSubdivs):
    control_points = np.array(control_points)
    control_points = control_points.reshape(control_points.shape[0], zSubdivs, -1, 3)

    knotvector_u = utilities.generate_knot_vector(3, control_points.shape[1])
    knotvector_v = utilities.generate_knot_vector(3, control_points.shape[2])

    # Get Framewise Conrol Points and Target Points
    frame_ctrl_pnts = control_points

    # Preliminaries for actual Optimization
    num_eval_pts_u = len(knotvector_u)
    num_eval_pts_v = len(knotvector_v)

    #frame_ctrl_pnts = np.expand_dims(frame_ctrl_pnts, 0)
    num_ctrl_pts1 = np.array(frame_ctrl_pnts.shape[1])
    num_ctrl_pts2 = np.array(frame_ctrl_pnts.shape[2])

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=3, q=3, u=None, v=None, out_dim_u=32, out_dim_v=128)

    inp_ctrl_pts = torch.FloatTensor(frame_ctrl_pnts.astype(np.float32))
    weights = torch.ones(inp_ctrl_pts.shape[0], num_ctrl_pts1, num_ctrl_pts2, 1)

    return layer(torch.cat((inp_ctrl_pts, weights), -1))


def compute_faces(U, V, inverted=False):
    faces = []

    if not inverted:
        # Regular triangulation
        for u in range(V - 1):
            for v in range(U - 1):
                faces.append(np.array([u*U + v, (u + 1)*U + v, (u+1)*U + v + 1], dtype=np.int))
                faces.append(np.array([u*U + v, (u + 1)*U + v + 1, (u)*U + v + 1], dtype=np.int))
        
        # Connect first to last row
        for v in range(V - 1):
            faces.append(np.array([U*v, U*v + U-1, U*(v+1)], dtype=np.int))
            faces.append(np.array([U*(v+1), U*v + U-1, U*(v+1) + U-1], dtype=np.int))

        # Add caps
        for u in range(U-2):
            faces.append(np.array([0, u, u+1], dtype=np.int))
            faces.append(np.array([(V-1)*U, (V-1)*U + u + 1, (V-1)*U + u], dtype=np.int))
    else:
        # Regular triangulation
        for u in range(V - 1):
            for v in range(U - 1):
                faces.append(np.array([u*U + v, (u+1)*U + v + 1, (u + 1)*U + v], dtype=np.int))
                faces.append(np.array([u*U + v, (u)*U + v + 1, (u + 1)*U + v + 1], dtype=np.int))
        
        # Connect first to last row
        for v in range(V - 1):
            faces.append(np.array([U*v, U*(v+1), U*v + U-1], dtype=np.int))
            faces.append(np.array([U*(v+1), U*(v+1) + U-1, U*v + U-1], dtype=np.int))

        # Add caps
        for u in range(U-2):
            faces.append(np.array([0, u+1, u], dtype=np.int))
            faces.append(np.array([(V-1)*U, (V-1)*U + u, (V-1)*U + u + 1], dtype=np.int))

    return np.array(faces)


def generate_BM5_mesh(control_points_left, control_points_right, zSubdivs):
    left_surface = generate_surface_points(control_points_left, zSubdivs).detach().cpu().numpy()
    right_surface = generate_surface_points(control_points_right, zSubdivs).detach().cpu().numpy()

    faces_left = compute_faces(left_surface.shape[2], left_surface.shape[1], inverted=False)
    faces_right = compute_faces(right_surface.shape[2], right_surface.shape[1], inverted=True)

    return left_surface, right_surface, faces_left, faces_right