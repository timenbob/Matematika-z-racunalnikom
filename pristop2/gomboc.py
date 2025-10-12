#!/usr/bin/env python3
import numpy as np
import trimesh as tm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import os

# ---------------------------
# Helpers
# ---------------------------
def to44(R3: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R3
    return T

def safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

# ---------------------------
# Utility: Fibonacci sampling on sphere
# ---------------------------
def fibonacci_sphere(n_dirs: int, randomize: bool = False, seed: int = 0):
    rng = np.random.RandomState(seed) if randomize else None
    rnd = rng.rand() * n_dirs if rng is not None else 1.0
    points = []
    offset = 2.0 / n_dirs
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_dirs):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(max(0.0, 1 - y*y))
        phi = ((i + rnd) % n_dirs) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        p = np.array([x, y, z], dtype=np.float64)
        p = safe_unit(p)
        points.append(p)
    return np.vstack(points)

# ---------------------------
# Mesh helpers
# ---------------------------
def load_mesh(path: str, force_convex: bool = True):
    mesh = tm.load_mesh(path, force='mesh')
    if not isinstance(mesh, tm.Trimesh):
        mesh = mesh.dump().sum()  # Scene -> single mesh

    # Čiščenje: nove API metode namesto deprecated klicev
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.process(validate=True)

    if force_convex and not mesh.is_convex:
        mesh = mesh.convex_hull

    # Normaliziraj merilo (povprečni radij ~ 1) in težišče v izhodišču
    v = mesh.vertices - mesh.center_mass
    rad = np.sqrt((v**2).sum(1)).mean()
    if rad > 0:
        s = 1.0 / rad
        mesh.apply_scale(s)
        mesh.apply_translation(-mesh.center_mass)
    return mesh

def ensure_watertight_convex(mesh: tm.Trimesh):
    if (not mesh.is_watertight) or (not mesh.is_convex):
        mesh = mesh.convex_hull
    return mesh

def oriented_mesh(mesh: tm.Trimesh, up_dir: np.ndarray):
    """
    Vrne kopijo mreže rotirano tako, da je up_dir poravnan z +Z.
    """
    up_dir = safe_unit(up_dir)
    z = np.array([0.0, 0.0, 1.0])

    if np.allclose(up_dir, z):
        rot = R.identity()
    elif np.allclose(up_dir, -z):
        rot = R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        v = np.cross(up_dir, z)
        s = np.linalg.norm(v)
        c = float(np.dot(up_dir, z))
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]], dtype=float)
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-16))
        rot = R.from_matrix(Rm)

    m2 = mesh.copy()
    m2.apply_transform(to44(rot.as_matrix()))
    return m2, rot

def place_on_plane(mesh: tm.Trimesh):
    """
    Premakni mrežo po z tako, da je najnižja točka na z=0 (nosilna ravnina).
    Vrne (kopijo, zmin, višina težišča).
    """
    zmin = float(mesh.vertices[:, 2].min())
    m2 = mesh.copy()
    m2.apply_translation([0, 0, -zmin])
    com_h = float(m2.center_mass[2])
    return m2, zmin, com_h

# ---------------------------
# Potential "energy" proxy: COM height when resting on plane
# ---------------------------
def com_height_for_updir(mesh: tm.Trimesh, up_dir: np.ndarray):
    mR, rot = oriented_mesh(mesh, up_dir)
    mPlaced, _, h = place_on_plane(mR)
    return h, mPlaced, rot

# ---------------------------
# Stability test via small tilts
# ---------------------------
def classify_orientation(mesh: tm.Trimesh, up_dir: np.ndarray, eps: float = 1e-3):
    """
    Vrne ('stable'|'unstable'|'saddle', base_height) za dano up-smer.
    """
    h0, m0, _ = com_height_for_updir(mesh, up_dir)

    # nagni okoli osi X in Y v orientiranem koordinatnem sistemu
    def tilt_height(axis):
        Rt = R.from_rotvec(eps * axis)
        m = m0.copy()
        m.apply_transform(to44(Rt.as_matrix()))
        mPlaced, _, h = place_on_plane(m)
        return h

    hx_p = tilt_height(np.array([1.0, 0.0, 0.0]))
    hx_m = tilt_height(np.array([-1.0, 0.0, 0.0]))
    hy_p = tilt_height(np.array([0.0, 1.0, 0.0]))
    hy_m = tilt_height(np.array([0.0, -1.0, 0.0]))

    # diskretni "drugi odvod" (končne razlike)
    dx2 = (hx_p + hx_m - 2*h0) / (eps**2)
    dy2 = (hy_p + hy_m - 2*h0) / (eps**2)

    tol = 1e-6
    posx = dx2 > tol
    posy = dy2 > tol
    negx = dx2 < -tol
    negy = dy2 < -tol

    if posx and posy:
        label = 'stable'       # lokalni minimum
    elif negx and negy:
        label = 'unstable'     # lokalni maksimum
    else:
        label = 'saddle'
    return label, h0

# ---------------------------
# Equilibria detection over sphere
# ---------------------------
def _cluster(indices, heights, dirs, pick='min'):
    """
    Sklopi bližnje smeri (≈ <5°) in izberi reprezentanta
    - pick='min'  -> argmin(h)
    - pick='max'  -> argmax(h)
    """
    if len(indices) == 0:
        return []
    taken = np.zeros(len(indices), dtype=bool)
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
            if float(np.dot(pa, pb)) > 0.995:  # ~ <5°
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

def detect_equilibria(mesh: tm.Trimesh, n_dirs=2000, eps=1e-3, seed=0):
    dirs = fibonacci_sphere(n_dirs, randomize=True, seed=seed)
    heights = np.zeros(n_dirs, dtype=float)
    for i, d in enumerate(dirs):
        h, _, _ = com_height_for_updir(mesh, d)
        heights[i] = h

    # sferični sosedje (približek)
    tree = cKDTree(dirs)
    k = 12
    nn = tree.query(dirs, k=k+1)[1][:, 1:]  # brez sebstiča

    labels = np.empty(n_dirs, dtype=object)
    for i in range(n_dirs):
        hi = heights[i]
        neigh = heights[nn[i]]
        is_min = np.all(hi <= neigh - 1e-8)
        is_max = np.all(hi >= neigh + 1e-8)
        if is_min or is_max:
            lab, _ = classify_orientation(mesh, dirs[i], eps=eps)
            labels[i] = lab
        else:
            labels[i] = None

    stable_raw   = [i for i, l in enumerate(labels) if l == 'stable']
    unstable_raw = [i for i, l in enumerate(labels) if l == 'unstable']
    saddle_raw   = [i for i, l in enumerate(labels) if l == 'saddle']

    stable_idx   = _cluster(stable_raw,   heights, dirs, pick='min')
    unstable_idx = _cluster(unstable_raw, heights, dirs, pick='max')
    saddle_idx   = _cluster(saddle_raw,   heights, dirs, pick='min')  # poljubno

    return {
        'dirs': dirs,
        'heights': heights,
        'stable_idx': stable_idx,
        'unstable_idx': unstable_idx,
        'saddle_idx': saddle_idx,
        'labels': labels
    }

# ---------------------------
# Geometry tweaks (very small)
# ---------------------------
def vertex_normals(mesh: tm.Trimesh):
    # v tej verziji so normalne property, ki se izračunajo on-demand
    return mesh.vertex_normals.copy()

def push_vertices_local(mesh: tm.Trimesh, target_dirs, amount=1e-3, sigma_deg=10.0, sign=+1):
    """
    Porini oglišča po normalah z Gaussovskimi težami okoli target up-smeri.
    sign=+1: razširi (dvigne h v tistih bazenih) -> ubije minima
    sign=-1: stisni (zniža h v tistih bazenih)   -> ubije maksima
    """
    verts = mesh.vertices.copy()
    norms = vertex_normals(mesh)

    c = mesh.center_mass
    vdir = verts - c
    vdir = (vdir.T / (np.linalg.norm(vdir, axis=1) + 1e-12)).T

    sigma = np.deg2rad(sigma_deg)
    total_w = np.zeros(len(verts), dtype=float)

    for d in target_dirs:
        d = safe_unit(np.asarray(d, dtype=float))
        cosang = np.clip(vdir @ d, -1.0, 1.0)
        ang = np.arccos(cosang)
        w = np.exp(-0.5 * (ang / (sigma + 1e-12))**2)
        total_w += w

    if total_w.max() > 0:
        total_w = total_w / total_w.max()

    disp = (sign * amount) * total_w[:, None] * norms
    verts_new = verts + disp

    new_mesh = mesh.copy()
    new_mesh.vertices = verts_new
    # invalidiraj cache za vsak slučaj
    if hasattr(new_mesh, "_cache"):
        new_mesh._cache.clear()

    new_mesh = ensure_watertight_convex(new_mesh)
    new_mesh.apply_translation(-new_mesh.center_mass)
    return new_mesh

def sphericize(mesh: tm.Trimesh, alpha=1e-3):
    """
    Rahlo zmešaj proti krogli okrog COM (glajenje).
    """
    verts = mesh.vertices.copy()
    c = mesh.center_mass
    u = verts - c
    r = np.linalg.norm(u, axis=1) + 1e-12
    r_mean = r.mean()
    target = c + (u / r[:, None]) * r_mean
    verts_new = (1 - alpha) * verts + alpha * target

    new_mesh = mesh.copy()
    new_mesh.vertices = verts_new
    if hasattr(new_mesh, "_cache"):
        new_mesh._cache.clear()

    new_mesh = ensure_watertight_convex(new_mesh)
    new_mesh.apply_translation(-new_mesh.center_mass)
    return new_mesh

# ---------------------------
# High-level loop
# ---------------------------
def evaluate(mesh, n_dirs=2000, eps=1e-3, verbose=True):
    eq = detect_equilibria(mesh, n_dirs=n_dirs, eps=eps)
    s = len(eq['stable_idx'])
    u = len(eq['unstable_idx'])
    t = len(eq['saddle_idx'])
    if verbose:
        print(f"Equilibria: stable={s}, unstable={u}, saddle={t}  (target: 1,1,0)")
    return eq

def tune_to_gomboc(mesh,
                   max_iters=50,
                   n_dirs=2000,
                   eps=1e-3,
                   local_push=5e-4,
                   sphere_alpha=5e-4,
                   save_every=1,
                   outdir="tuning_out"):
    os.makedirs(outdir, exist_ok=True)
    m = mesh.copy()

    for it in range(1, max_iters+1):
        print(f"\n=== Iteration {it} ===")
        eq = evaluate(m, n_dirs=n_dirs, eps=eps, verbose=True)

        s, u, t = len(eq['stable_idx']), len(eq['unstable_idx']), len(eq['saddle_idx'])
        if s == 1 and u == 1:
            print("Potential Gömböc candidate reached (by sampled test).")
            tm.exchange.export.export_mesh(m, os.path.join(outdir, f"candidate_it{it}.stl"))
            break

        # preveč stabilnih minimumov -> razširi okolico "odvečnih"
        if s > 1 and len(eq['stable_idx']) > 0:
            extra = eq['stable_idx'][1:]  # prvega pusti pri miru
            dirs = [eq['dirs'][i] for i in extra]
            m = push_vertices_local(m, dirs, amount=local_push, sigma_deg=12.0, sign=+1)

        # preveč nestabilnih maksimumov -> stisni okolico "odvečnih"
        if u > 1 and len(eq['unstable_idx']) > 0:
            extra = eq['unstable_idx'][1:]
            dirs = [eq['dirs'][i] for i in extra]
            m = push_vertices_local(m, dirs, amount=local_push, sigma_deg=12.0, sign=-1)

        # če je sedel, malo zgladi
        if t > 0:
            m = sphericize(m, alpha=sphere_alpha)

        # ohrani konveksnost in COM v 0
        m = ensure_watertight_convex(m)
        m.apply_translation(-m.center_mass)

        if save_every and (it % save_every == 0):
            tm.exchange.export.export_mesh(m, os.path.join(outdir, f"mesh_it{it}.stl"))

    return m

# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=False,
                    default="/home/timen/Documents/Faks/Matematika-z-racunalnikom/pristop2/gomboc(1).stl",
                    help="Pot do STL datoteke")
    ap.add_argument("--iters", type=int, default=33000)
    ap.add_argument("--dirs", type=int, default=40000)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--out", type=str, default="tuning_out")
    args = ap.parse_args()

    mesh0 = load_mesh(args.stl, force_convex=True)
    tuned = tune_to_gomboc(mesh0,
                           max_iters=args.iters,
                           n_dirs=args.dirs,
                           eps=args.eps,
                           outdir=args.out)

    out_final = os.path.join(args.out, "final_candidate.stl")
    tm.exchange.export.export_mesh(tuned, out_final)
    print("Saved:", out_final)
