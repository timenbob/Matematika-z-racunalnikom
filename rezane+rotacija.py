#!/usr/bin/env python3
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def to44(R3):
    T = np.eye(4)
    T[:3, :3] = R3
    return T

def safe_unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def reflect_points_across_plane(points, n):
    n = safe_unit(n)
    d = points @ n
    return points - 2.0 * d[:, None] * n[None, :]


def _mirror_score(V, normals):
    """How symmetric V is w.r.t. given normals."""
    tree = cKDTree(V)
    score = 0.0
    for n in normals:
        Vref = reflect_points_across_plane(V, n)
        d, _ = tree.query(Vref, k=1)
        score += float((d**2).mean())
    return score


# --------------------------------------------------------
# MAIN AUTO-ORIENT FUNCTION
# --------------------------------------------------------
def auto_orient_to_two_mirrors(mesh, coarse_deg=2):
    """
    Rotates STL so that it has symmetry planes:
       x = 0  and  y = 0
    """
    mesh = mesh.copy()
    mesh.vertices -= mesh.center_mass
    
    # PCA-based alignment
    I = mesh.moment_inertia
    evals, evecs = np.linalg.eigh(I)
    P = evecs[:, np.argsort(evals)]
    if np.linalg.det(P) < 0:
        P[:, 0] *= -1

    mesh.apply_transform(to44(P.T))
    V = mesh.vertices.copy()

    normals = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    best = (np.inf, 0)

    # Search best rotation around Z
    for deg in range(0, 360, coarse_deg):
        Rz = R.from_euler('z', deg, degrees=True).as_matrix()
        Vr = V @ Rz.T
        sc = _mirror_score(Vr, normals)
        if sc < best[0]:
            best = (sc, deg)

    R_final = R.from_euler('z', best[1], degrees=True).as_matrix()
    mesh.apply_transform(to44(R_final))
    mesh.vertices -= mesh.center_mass

    return mesh


# --------------------------------------------------------
# CUT MESH ALONG Y = 0
# --------------------------------------------------------
def slice_mesh_y0(mesh):
    """
    Cuts mesh into left (y<0) and right (y>0) halves.
    Returns (left_mesh, right_mesh)
    """
    plane_origin = np.array([0, 0, 0])
    plane_normal = np.array([0, 0, 1])  # y-axis normal

    # trimesh slicing:
    left = mesh.slice_plane(plane_origin, plane_normal)
    right = mesh.slice_plane(plane_origin, -plane_normal)

    return left, right


# --------------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------------
if __name__ == "__main__":

    stl_path = "/home/timen/Documents/Faks/Matematika-z-racunalnikom/koncni-kandidati/Monostatic_Body_Release.STL"

    mesh = trimesh.load(stl_path, force="mesh")

    # Auto-orient
    mesh = auto_orient_to_two_mirrors(mesh)

    # Move COM to origin
    mesh.apply_translation(-mesh.center_mass)

    # Slice
    left_mesh, right_mesh = slice_mesh_y0(mesh)

    left_mesh.export("gomboc_left.stl")
    right_mesh.export("gomboc_right.stl")

    print("Rezanje zaključeno.")
