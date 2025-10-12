#!/usr/bin/env python3
import numpy as np
import trimesh as tm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import os
from itertools import permutations, product

# =========================
# Helpers
# =========================
def to44(R3: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R3
    return T

def safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

def principal_axes_R(mesh: tm.Trimesh) -> np.ndarray:
    """3x3 ortonormalna matrika P (stolpci so glavne osi; desnosučna baza)."""
    I = mesh.moment_inertia
    evals, evecs = np.linalg.eigh(I)
    P = evecs[:, np.argsort(evals)]
    if np.linalg.det(P) < 0:
        P[:, 2] *= -1.0
    return P

# =========================
# Sfera (vzorcevanje)
# =========================
def fibonacci_sphere(n_dirs: int, randomize: bool = False, seed: int = 0):
    rng = np.random.RandomState(seed) if randomize else None
    rnd = rng.rand() * n_dirs if rng is not None else 1.0
    points = []
    offset = 2.0 / n_dirs
    inc = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_dirs):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(max(0.0, 1 - y*y))
        phi = ((i + rnd) % n_dirs) * inc
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append(safe_unit(np.array([x, y, z], dtype=np.float64)))
    return np.vstack(points)

# =========================
# Mesh helpers
# =========================
def load_mesh(path: str, force_convex: bool = True):
    mesh = tm.load_mesh(path, force='mesh')
    if not isinstance(mesh, tm.Trimesh):
        mesh = mesh.dump().sum()
    # Čiščenje (brez deprecated klicev)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.process(validate=True)
    if force_convex and not mesh.is_convex:
        mesh = mesh.convex_hull
    # Normalizacija: povpr. radij ~ 1, COM v 0
    v = mesh.vertices - mesh.center_mass
    rad = np.sqrt((v**2).sum(1)).mean()
    if rad > 0:
        mesh.apply_scale(1.0 / rad)
        mesh.apply_translation(-mesh.center_mass)
    return mesh

def ensure_watertight_convex(mesh: tm.Trimesh):
    if (not mesh.is_watertight) or (not mesh.is_convex):
        mesh = mesh.convex_hull
    return mesh

# =========================
# Merilo simetrije + auto-orient
# =========================
def reflect_points_across_plane(points: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = safe_unit(n)
    dot = points @ n
    return points - 2.0 * dot[:, None] * n[None, :]

def _mirror_score(V, normals):
    """Manjše je boljše: MSE razdalje do najbližjih točk po zrcaljenju čez dane ravnine."""
    tree = cKDTree(V)
    score = 0.0
    for n in normals:
        Vref = reflect_points_across_plane(V, n)
        d, _ = tree.query(Vref, k=1)
        score += float((d**2).mean())
    return score

def auto_orient_to_two_mirrors(mesh: tm.Trimesh, coarse_deg=5):
    """
    Najdi rotacijo R, ki najbolj poravna objekt tako,
    da sta ravnini simetrije poravnani na world X=0 in Y=0.
    (Normalama [1,0,0] in [0,1,0].)
    """
    P = principal_axes_R(mesh)  # part-osi v world
    V0 = mesh.vertices.copy()
    normals_world_xy = [np.array([1.,0.,0.]), np.array([0.,1.,0.])]
    best = (np.inf, np.eye(3))
    # vse desnosučne permutacije P + vrtenje okoli Z
    for perm in permutations([0,1,2]):
        B = P[:, perm]
        for signs in product([1.,-1.],[1.,-1.],[1.,-1.]):
            R0 = B @ np.diag(signs)
            if np.linalg.det(R0) < 0:
                continue
            for deg in range(0, 360, coarse_deg):
                Rz = R.from_euler('z', deg, degrees=True).as_matrix()
                Rtry = Rz @ R0
                V = (V0 @ Rtry.T) - (V0 @ Rtry.T).mean(axis=0)
                sc = _mirror_score(V, normals_world_xy)
                if sc < best[0]:
                    best = (sc, Rtry)
    Rbest = best[1]
    m = mesh.copy()
    m.apply_transform(to44(Rbest))
    m.apply_translation(-m.center_mass)
    return m, Rbest

# =========================
# Potencial ~ višina COM
# =========================
def oriented_mesh(mesh: tm.Trimesh, up_dir: np.ndarray):
    up_dir = safe_unit(up_dir)
    z = np.array([0.,0.,1.])
    if np.allclose(up_dir, z):
        rot = R.identity()
    elif np.allclose(up_dir, -z):
        rot = R.from_rotvec(np.pi * np.array([1.,0.,0.]))
    else:
        v = np.cross(up_dir, z); s = np.linalg.norm(v); c = float(np.dot(up_dir, z))
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=float)
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-16))
        rot = R.from_matrix(Rm)
    m2 = mesh.copy()
    m2.apply_transform(to44(rot.as_matrix()))
    return m2, rot

def place_on_plane(mesh: tm.Trimesh):
    zmin = float(mesh.vertices[:,2].min())
    m2 = mesh.copy()
    m2.apply_translation([0,0,-zmin])
    com_h = float(m2.center_mass[2])
    return m2, zmin, com_h

def com_height_for_updir(mesh: tm.Trimesh, up_dir: np.ndarray):
    mR, _ = oriented_mesh(mesh, up_dir)
    mPlaced, _, h = place_on_plane(mR)
    return h, mPlaced

# =========================
# Klasifikacija (min/max/sedlo)
# =========================
def classify_orientation(mesh: tm.Trimesh, up_dir: np.ndarray, eps: float = 1e-3):
    h0, m0 = com_height_for_updir(mesh, up_dir)
    def tilt_height(axis):
        Rt = R.from_rotvec(eps * axis)
        m = m0.copy()
        m.apply_transform(to44(Rt.as_matrix()))
        mP, _, h = place_on_plane(m)
        return h
    hx_p = tilt_height(np.array([ 1.0, 0.0, 0.0]))
    hx_m = tilt_height(np.array([-1.0, 0.0, 0.0]))
    hy_p = tilt_height(np.array([ 0.0, 1.0, 0.0]))
    hy_m = tilt_height(np.array([ 0.0,-1.0, 0.0]))
    dx2 = (hx_p + hx_m - 2*h0) / (eps**2)
    dy2 = (hy_p + hy_m - 2*h0) / (eps**2)
    tol = 1e-6
    posx, posy = dx2 > tol, dy2 > tol
    negx, negy = dx2 < -tol, dy2 < -tol
    if posx and posy:   return 'stable',   h0
    if negx and negy:   return 'unstable', h0
    return 'saddle', h0

# =========================
# Detekcija ekvilibrijev
# =========================
def _cluster(indices, heights, dirs, pick='min'):
    if len(indices) == 0:
        return []
    taken = np.zeros(len(indices), dtype=bool)
    groups = []
    for a in range(len(indices)):
        if taken[a]: continue
        ia = indices[a]; pa = dirs[ia]; group = [ia]
        for b in range(a + 1, len(indices)):
            if taken[b]: continue
            ib = indices[b]; pb = dirs[ib]
            if float(np.dot(pa, pb)) > 0.995:
                taken[b] = True; group.append(ib)
        taken[a] = True; groups.append(group)
    reps = []
    for g in groups:
        hs = heights[g]
        reps.append(g[int(np.argmin(hs))] if pick == 'min' else g[int(np.argmax(hs))])
    return reps

def detect_equilibria(mesh: tm.Trimesh, n_dirs=2000, eps=1e-3, seed=0):
    dirs = fibonacci_sphere(n_dirs, randomize=True, seed=seed)
    heights = np.zeros(n_dirs, dtype=float)
    for i, d in enumerate(dirs):
        h, _ = com_height_for_updir(mesh, d)
        heights[i] = h
    tree = cKDTree(dirs); k = 12
    nn = tree.query(dirs, k=k+1)[1][:, 1:]
    labels = np.empty(n_dirs, dtype=object)
    for i in range(n_dirs):
        hi = heights[i]; neigh = heights[nn[i]]
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
    saddle_idx   = _cluster(saddle_raw,   heights, dirs, pick='min')
    return {'dirs':dirs,'heights':heights,
            'stable_idx':stable_idx,'unstable_idx':unstable_idx,
            'saddle_idx':saddle_idx,'labels':labels}

# =========================
# Geometrijski popravki + simetrije
# =========================
def vertex_normals(mesh: tm.Trimesh):
    return mesh.vertex_normals.copy()

def push_vertices_local(mesh: tm.Trimesh, target_dirs, amount=1e-3, sigma_deg=10.0, sign=+1):
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
    if hasattr(new_mesh, "_cache"): new_mesh._cache.clear()
    new_mesh = ensure_watertight_convex(new_mesh)
    new_mesh.apply_translation(-new_mesh.center_mass)
    return new_mesh

def sphericize(mesh: tm.Trimesh, alpha=1e-3):
    verts = mesh.vertices.copy()
    c = mesh.center_mass
    u = verts - c
    r = np.linalg.norm(u, axis=1) + 1e-12
    r_mean = r.mean()
    target = c + (u / r[:, None]) * r_mean
    verts_new = (1 - alpha) * verts + alpha * target
    new_mesh = mesh.copy()
    new_mesh.vertices = verts_new
    if hasattr(new_mesh, "_cache"): new_mesh._cache.clear()
    new_mesh = ensure_watertight_convex(new_mesh)
    new_mesh.apply_translation(-new_mesh.center_mass)
    return new_mesh

def enforce_mirror(mesh: tm.Trimesh, normal: np.ndarray, k=1):
    normal = safe_unit(normal)
    V = mesh.vertices.copy()
    V_m = reflect_points_across_plane(V, normal)
    tree = cKDTree(V)
    _, idx = tree.query(V_m, k=k)
    idx = idx[:, 0] if k > 1 else idx
    newV = V.copy()
    for i, j in enumerate(idx):
        vi = newV[i]; vj = newV[j]
        vi_new = 0.5 * (vi + reflect_points_across_plane(vj[None, :], normal)[0])
        vj_new = reflect_points_across_plane(vi_new[None, :], normal)[0]
        newV[i] = vi_new; newV[j] = vj_new
    mesh.vertices = newV
    if hasattr(mesh, "_cache"): mesh._cache.clear()
    return mesh

def break_plane_mirror(mesh: tm.Trimesh, normal: np.ndarray, amount=1e-4):
    """Namenoma razbij strogo zrcalo v ravnini n·x=0 (mikro antisimetričen premik)."""
    normal = safe_unit(normal)
    V = mesh.vertices.copy()
    norms = mesh.vertex_normals
    sgn = np.sign(V @ normal)[:, None]
    V_new = V + amount * sgn * norms
    new_mesh = mesh.copy()
    new_mesh.vertices = V_new
    if hasattr(new_mesh, "_cache"): new_mesh._cache.clear()
    new_mesh = ensure_watertight_convex(new_mesh)
    new_mesh.apply_translation(-new_mesh.center_mass)
    return new_mesh

def ensure_exactly_two_mirrors(mesh: tm.Trimesh,
                               n1: np.ndarray = np.array([1.,0.,0.]),
                               n2: np.ndarray = np.array([0.,1.,0.]),
                               break_normal: np.ndarray = np.array([0.,0.,1.]),
                               break_amount=1e-4):
    """
    Strogo uveljavi le 2 zrcali (n1, n2) in razbij morebitno zrcalo čez break_normal.
    Privzeto: n1=X, n2=Y, break=Z (po auto-orientu to pomeni “simetriji prvotnega objekta”).
    """
    m = mesh.copy()
    m = enforce_mirror(m, n1)
    m = enforce_mirror(m, n2)
    m = break_plane_mirror(m, break_normal, amount=break_amount)
    m = ensure_watertight_convex(m)
    m.apply_translation(-m.center_mass)
    return m

# =========================
# Zanka
# =========================
def evaluate(mesh, n_dirs=2000, eps=1e-3, verbose=True):
    eq = detect_equilibria(mesh, n_dirs=n_dirs, eps=eps)
    s = len(eq['stable_idx']); u = len(eq['unstable_idx']); t = len(eq['saddle_idx'])
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
                   outdir="tuning_out",
                   break_amount=1e-4):
    os.makedirs(outdir, exist_ok=True)
    m = mesh.copy()

    # 1) Auto-orient: poravnaj objekt, da njegove 2 simetriji prideta na X=0, Y=0 (world)
    m, _ = auto_orient_to_two_mirrors(m)

    # 2) Zakleni natanko ti dve (in razbij tretjo Z)
    m = ensure_exactly_two_mirrors(m, break_amount=break_amount)

    for it in range(1, max_iters+1):
        print(f"\n=== Iteration {it} ===")
        eq = evaluate(m, n_dirs=n_dirs, eps=eps, verbose=True)

        s, u, t = len(eq['stable_idx']), len(eq['unstable_idx']), len(eq['saddle_idx'])
        if s == 1 and u == 1:
            print("Potential Gömböc candidate reached (by sampled test).")
            tm.exchange.export.export_mesh(m, os.path.join(outdir, f"candidate_it{it}.stl"))
            break

        if s > 1 and len(eq['stable_idx']) > 0:
            extra = eq['stable_idx'][1:]
            dirs = [eq['dirs'][i] for i in extra]
            m = push_vertices_local(m, dirs, amount=local_push, sigma_deg=12.0, sign=+1)

        if u > 1 and len(eq['unstable_idx']) > 0:
            extra = eq['unstable_idx'][1:]
            dirs = [eq['dirs'][i] for i in extra]
            m = push_vertices_local(m, dirs, amount=local_push, sigma_deg=12.0, sign=-1)

        if t > 0:
            m = sphericize(m, alpha=sphere_alpha)

        # 3) Po vsakem koraku ohrani NATAČNO isti 2 ravnini (X,Y)
        m = ensure_exactly_two_mirrors(m, break_amount=break_amount)

        m = ensure_watertight_convex(m)
        m.apply_translation(-m.center_mass)

        if save_every and (it % save_every == 0):
            tm.exchange.export.export_mesh(m, os.path.join(outdir, f"mesh_it{it}.stl"))

    return m

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=False,
                    default="/home/timen/Documents/Faks/Matematika-z-racunalnikom/pristop2/gomboc(1).stl",
                    help="Pot do STL datoteke")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--dirs", type=int, default=6000)
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--out", type=str, default="tuning_out")
    ap.add_argument("--break_amount", type=float, default=1e-4,
                    help="Mikro-razbitje morebitne tretje zrcalne ravnine")
    args = ap.parse_args()

    mesh0 = load_mesh(args.stl, force_convex=True)

    tuned = tune_to_gomboc(mesh0,
                           max_iters=args.iters,
                           n_dirs=args.dirs,
                           eps=args.eps,
                           outdir=args.out,
                           break_amount=args.break_amount)

    out_final = os.path.join(args.out, "final_candidate.stl")
    tm.exchange.export.export_mesh(tuned, out_final)
    print("Saved:", out_final)
