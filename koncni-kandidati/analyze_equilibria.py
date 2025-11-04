#!/usr/bin/env python3
import numpy as np
import trimesh as tm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import os

# --------------------------- Helpers ---------------------------
def to44(R3): 
    T = np.eye(4); T[:3, :3] = R3
    return T

def safe_unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)

def fibonacci_sphere(n_dirs=2000, seed=0):
    rng = np.random.RandomState(seed)
    rnd = rng.rand() * n_dirs
    pts = []
    offset = 2.0 / n_dirs
    inc = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_dirs):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(max(0.0, 1 - y*y))
        phi = ((i + rnd) % n_dirs) * inc
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        pts.append(np.array([x, y, z]))
    return np.array(pts)

# --------------------------- Mesh helpers ---------------------------
def load_mesh(path):
    mesh = tm.load_mesh(path, force='mesh')
    if not isinstance(mesh, tm.Trimesh):
        mesh = mesh.dump().sum()
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.process(validate=True)
    mesh.apply_translation(-mesh.center_mass)
    return mesh

def oriented_mesh(mesh, up_dir):
    up_dir = safe_unit(up_dir)
    z = np.array([0, 0, 1.0])
    if np.allclose(up_dir, z):
        rot = R.identity()
    elif np.allclose(up_dir, -z):
        rot = R.from_rotvec(np.pi * np.array([1, 0, 0]))
    else:
        v = np.cross(up_dir, z)
        s = np.linalg.norm(v)
        c = float(np.dot(up_dir, z))
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]], float)
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-16))
        rot = R.from_matrix(Rm)
    m2 = mesh.copy()
    m2.apply_transform(to44(rot.as_matrix()))
    return m2

def place_on_plane(mesh):
    zmin = mesh.vertices[:, 2].min()
    m2 = mesh.copy()
    m2.apply_translation([0, 0, -zmin])
    return m2, float(m2.center_mass[2])

def com_height_for_updir(mesh, up_dir):
    mR = oriented_mesh(mesh, up_dir)
    _, h = place_on_plane(mR)
    return h

def classify_orientation(mesh, up_dir, eps=1e-3):
    h0 = com_height_for_updir(mesh, up_dir)

    def tilt(ax):
        ax = np.array(ax, dtype=float)
        Rt = R.from_rotvec(eps * ax)
        m = oriented_mesh(mesh, up_dir)
        m.apply_transform(to44(Rt.as_matrix()))
        _, h = place_on_plane(m)
        return h

    hx_p = tilt([1, 0, 0])
    hx_m = tilt([-1, 0, 0])
    hy_p = tilt([0, 1, 0])
    hy_m = tilt([0, -1, 0])

    dx2 = (hx_p + hx_m - 2 * h0) / (eps**2)
    dy2 = (hy_p + hy_m - 2 * h0) / (eps**2)
    tol = 1e-6
    posx, posy = dx2 > tol, dy2 > tol
    negx, negy = dx2 < -tol, dy2 < -tol
    if posx and posy:
        return 'stable', h0
    if negx and negy:
        return 'unstable', h0
    return 'saddle', h0

def _cluster(indices, heights, dirs, pick='min'):
    if not indices:
        return []
    taken = np.zeros(len(indices), bool)
    groups = []
    for a in range(len(indices)):
        if taken[a]:
            continue
        ia = indices[a]
        pa = dirs[ia]
        group = [ia]
        for b in range(a + 1, len(indices)):
            if taken[b]:
                continue
            ib = indices[b]
            pb = dirs[ib]
            if float(np.dot(pa, pb)) > 0.995:
                taken[b] = True
                group.append(ib)
        taken[a] = True
        groups.append(group)
    reps = []
    for g in groups:
        hs = heights[g]
        rep = g[int(np.argmin(hs))] if pick == 'min' else g[int(np.argmax(hs))]
        reps.append(rep)
    return reps

def detect_equilibria(mesh, n_dirs=4000, eps=1e-3, seed=0):
    dirs = fibonacci_sphere(n_dirs, seed=seed)
    heights = np.zeros(n_dirs)
    for i, d in enumerate(dirs):
        heights[i] = com_height_for_updir(mesh, d)
    tree = cKDTree(dirs)
    nn = tree.query(dirs, k=13)[1][:, 1:]
    labels = np.empty(n_dirs, dtype=object)
    for i in range(n_dirs):
        hi = heights[i]
        neigh = heights[nn[i]]
        is_min = np.all(hi <= neigh - 1e-8)
        is_max = np.all(hi >= neigh + 1e-8)
        if is_min or is_max:
            lab, _ = classify_orientation(mesh, dirs[i], eps)
            labels[i] = lab
        else:
            labels[i] = None
    stable_idx = [i for i, l in enumerate(labels) if l == 'stable']
    unstable_idx = [i for i, l in enumerate(labels) if l == 'unstable']
    saddle_idx = [i for i, l in enumerate(labels) if l == 'saddle']
    return {
        'dirs': dirs,
        'heights': heights,
        'stable_idx': _cluster(stable_idx, heights, dirs, 'min'),
        'unstable_idx': _cluster(unstable_idx, heights, dirs, 'max'),
        'saddle_idx': _cluster(saddle_idx, heights, dirs, 'min')
    }

# --------------------------- Visualization ---------------------------
def save_equilibria_markers(mesh, eq, out_path):
    colors = []
    dirs = eq['dirs']
    verts = []
    for i in eq['stable_idx']:
        verts.append(dirs[i])
        colors.append([0, 0, 1])  # modra
    for i in eq['unstable_idx']:
        verts.append(dirs[i])
        colors.append([1, 0, 0])  # rdeča
    for i in eq['saddle_idx']:
        verts.append(dirs[i])
        colors.append([0, 1, 0])  # zelena

    if not verts:
        print("⚠️ No equilibria detected for visualization.")
        return

    verts = np.array(verts)
    colors = np.array(colors)
    pts = tm.points.PointCloud(verts, colors)
    scene = tm.Scene([mesh, pts])
    scene.export(out_path)
    print(f"💾 Saved visualization: {out_path}")

# --------------------------- Main CLI ---------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Analyze equilibria of a 3D STL body")
    ap.add_argument("--stl", required=True, help="Input STL file")
    ap.add_argument("--out", default="equilibria_marked.stl", help="Output STL with markers")
    ap.add_argument("--dirs", type=int, default=8000, help="Number of sample directions")
    ap.add_argument("--eps", type=float, default=1e-3, help="Small rotation for testing stability")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mesh = load_mesh(args.stl)
    eq = detect_equilibria(mesh, n_dirs=args.dirs, eps=args.eps, seed=args.seed)

    s, u, t = len(eq['stable_idx']), len(eq['unstable_idx']), len(eq['saddle_idx'])
    print(f"Stabilne lege:   {s}")
    print(f"Nestabilne lege: {u}")
    print(f"Sedlaste lege:   {t}")
    print(f"Vsota: {s + u + t}")

    save_equilibria_markers(mesh, eq, args.out)
