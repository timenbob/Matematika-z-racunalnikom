#!/usr/bin/env python3
import os, math, random
import numpy as np
import trimesh as tm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from itertools import permutations, product

# =========================
# Utility helpers
# =========================
def to44(R3: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R3
    return T

def safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

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

def principal_axes_R(mesh: tm.Trimesh) -> np.ndarray:
    I = mesh.moment_inertia
    evals, evecs = np.linalg.eigh(I)
    P = evecs[:, np.argsort(evals)]
    if np.linalg.det(P) < 0:
        P[:, 2] *= -1.0
    return P

# =========================
# Mesh I/O + normalization
# =========================
def load_mesh(path: str, force_convex: bool = True):
    mesh = tm.load_mesh(path, force='mesh')
    if not isinstance(mesh, tm.Trimesh):
        mesh = mesh.dump().sum()
    # cleanup z novimi API
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.process(validate=True)
    if force_convex and not mesh.is_convex:
        mesh = mesh.convex_hull
    # scale: povpr. radij ~ 1; COM v 0
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
# Auto-orient: poravnaj na 2 zrcali (x=0,y=0)
# =========================
def reflect_points_across_plane(points: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = safe_unit(n)
    dot = points @ n
    return points - 2.0 * dot[:, None] * n[None, :]

def _mirror_score(V, normals):
    tree = cKDTree(V)
    score = 0.0
    for n in normals:
        Vref = reflect_points_across_plane(V, n)
        d, _ = tree.query(Vref, k=1)
        score += float((d**2).mean())
    return score

def auto_orient_to_two_mirrors(mesh: tm.Trimesh, coarse_deg=5):
    P = principal_axes_R(mesh)  # part axes in world
    V0 = mesh.vertices.copy()
    normals_world_xy = [np.array([1.,0.,0.]), np.array([0.,1.,0.])]
    best = (np.inf, np.eye(3))
    for perm in permutations([0,1,2]):
        B = P[:, perm]
        for signs in product([1.,-1.],[1.,-1.],[1.,-1.]):
            R0 = B @ np.diag(signs)
            if np.linalg.det(R0) < 0:
                continue
            for deg in range(0, 360, coarse_deg):
                Rz = R.from_euler('z', deg, degrees=True).as_matrix()
                Rtry = Rz @ R0
                V = (V0 @ Rtry.T)
                V = V - V.mean(axis=0)
                sc = _mirror_score(V, normals_world_xy)
                if sc < best[0]:
                    best = (sc, Rtry)
    Rbest = best[1]
    m = mesh.copy()
    m.apply_transform(to44(Rbest))
    m.apply_translation(-m.center_mass)
    return m

# =========================
# Potential ~ COM height
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
    return m2

def place_on_plane(mesh: tm.Trimesh):
    zmin = float(mesh.vertices[:,2].min())
    m2 = mesh.copy()
    m2.apply_translation([0,0,-zmin])
    com_h = float(m2.center_mass[2])
    return m2, zmin, com_h

def com_height_for_updir(mesh: tm.Trimesh, up_dir: np.ndarray):
    mR = oriented_mesh(mesh, up_dir)
    mPlaced, _, h = place_on_plane(mR)
    return h

def classify_orientation(mesh: tm.Trimesh, up_dir: np.ndarray, eps: float = 1e-3):
    h0 = com_height_for_updir(mesh, up_dir)
    def tilt_height(axis):
        Rt = R.from_rotvec(eps * axis)
        m = oriented_mesh(mesh, up_dir)
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
        heights[i] = com_height_for_updir(mesh, d)
    tree = cKDTree(dirs)
    k = 12
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
# Symmetry: exactly two mirrors (X and Y) + break Z
# =========================
def vertex_normals(mesh: tm.Trimesh):
    return mesh.vertex_normals.copy()

def enforce_mirror(mesh: tm.Trimesh, normal: np.ndarray, k=1):
    normal = safe_unit(normal)
    V = mesh.vertices.copy()
    V_m = reflect_points_across_plane(V, normal)
    tree = cKDTree(V)
    _, idx = tree.query(V_m, k=k)
    idx = idx[:,0] if k>1 else idx
    newV = V.copy()
    for i, j in enumerate(idx):
        vi = newV[i]; vj = newV[j]
        vi_new = 0.5 * (vi + reflect_points_across_plane(vj[None,:], normal)[0])
        vj_new = reflect_points_across_plane(vi_new[None,:], normal)[0]
        newV[i] = vi_new; newV[j] = vj_new
    mesh.vertices = newV
    if hasattr(mesh, "_cache"): mesh._cache.clear()
    return mesh

def break_plane_mirror(mesh: tm.Trimesh, normal: np.ndarray, amount=1e-4):
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

def ensure_exactly_two_mirrors(mesh: tm.Trimesh, break_amount=1e-4):
    m = mesh.copy()
    m = enforce_mirror(m, np.array([1.,0.,0.]))
    m = enforce_mirror(m, np.array([0.,1.,0.]))
    m = break_plane_mirror(m, np.array([0.,0.,1.]), amount=break_amount)
    m = ensure_watertight_convex(m)
    m.apply_translation(-m.center_mass)
    return m

# =========================
# Deformation field: smooth RBF over sphere directions
# =========================
class SmoothDeformer:
    """
    Displaces vertices along their normals with a smooth, low-dim field:
      disp(p) = sum_k coeff[k] * exp(-0.5 * (angle(dir(p), cdir[k]) / sigma)^2)
    where dir(p) is the direction of vertex from COM.
    """
    def __init__(self, mesh: tm.Trimesh, K=32, sigma_deg=18.0, seed=0):
        self.K = K
        self.centers = fibonacci_sphere(K, randomize=True, seed=seed)  # control directions
        self.sigma = np.deg2rad(sigma_deg)
        self.base_vertices = mesh.vertices.copy()
        self.base_normals  = vertex_normals(mesh)
        self.base_com      = mesh.center_mass.copy()

    def apply(self, mesh: tm.Trimesh, coeff: np.ndarray, amount_scale=1.0):
        verts = self.base_vertices.copy()
        norms = self.base_normals.copy()
        # direction from COM
        vdir = verts - self.base_com
        vdir = (vdir.T / (np.linalg.norm(vdir, axis=1) + 1e-12)).T

        total = np.zeros(len(verts))
        for k in range(self.K):
            d = self.centers[k]
            cosang = np.clip(vdir @ d, -1.0, 1.0)
            ang = np.arccos(cosang)
            w = np.exp(-0.5 * (ang / (self.sigma + 1e-12))**2)
            total += coeff[k] * w

        disp = (amount_scale * total)[:,None] * norms
        newV = verts + disp
        mesh.vertices = newV
        if hasattr(mesh, "_cache"): mesh._cache.clear()
        return mesh

# =========================
# Fitness (loss)
# =========================
def fitness(mesh: tm.Trimesh, n_dirs=2000, eps=1e-3, seed=0,
            w_s=3.0, w_u=2.0, w_t=0.2):
    eq = detect_equilibria(mesh, n_dirs=n_dirs, eps=eps, seed=seed)
    s = len(eq['stable_idx'])
    u = len(eq['unstable_idx'])
    t = len(eq['saddle_idx'])
    # cilj: s=1, u=1, t ~ 0
    L = w_s * abs(s - 1) + w_u * abs(u - 1) + w_t * t
    return L, (s,u,t)

# =========================
# ES / NES optimizacija
# =========================
def optimize_shape(mesh_in: tm.Trimesh,
                   iters=300, pop=32, elite=6,
                   K=32, sigma_deg=18.0,
                   step_coeff=1e-3,   # max velikost geometrijske spremembe
                   coef_sigma=0.2,    # standard dev za koeficiente
                   n_dirs=2000, eps=1e-3, seed=0,
                   outdir="ml_out"):
    os.makedirs(outdir, exist_ok=True)

    # 0) auto-orient + fiksiraj dve zrcali
    m0 = auto_orient_to_two_mirrors(mesh_in)
    m0 = ensure_exactly_two_mirrors(m0, break_amount=1e-4)
    tm.exchange.export.export_mesh(m0, os.path.join(outdir, "00_auto_oriented.stl"))

    # 1) pripravi deformer (nizek dimenzijski param. prostor)
    deformer = SmoothDeformer(m0, K=K, sigma_deg=sigma_deg, seed=seed)

    # 2) NES: povprečje koeficientov + adaptivno
    rng = np.random.RandomState(seed)
    mu = np.zeros(K, dtype=float)
    sigma = np.ones(K, dtype=float) * coef_sigma

    best_mesh = m0.copy()
    best_score, best_counts = fitness(best_mesh, n_dirs=n_dirs, eps=eps, seed=seed)
    print(f"[init] L={best_score:.3f}  (s,u,t)={best_counts}")

    for it in range(1, iters+1):
        candidates = []
        # vzorči populacijo
        for j in range(pop):
            z = rng.randn(K)
            coeff = mu + sigma * z

            # sklop simetrije koeficientov: zrcaljenje X in Y -> uparjanje control directions
            # (enostaven trik: zrcali center directions in povpreči koef. mapirane pare)
            # Za preprostost: tukaj bomo striktno uveljavili simetrije na geometriji kasneje.

            # ustvarimo kandidatni mesh
            m = m0.copy()
            m = deformer.apply(m, coeff, amount_scale=step_coeff)
            # strogo ohrani 2 zrcali + razbij Z, konveksnost & COM
            m = ensure_exactly_two_mirrors(m, break_amount=1e-4)

            L, counts = fitness(m, n_dirs=n_dirs, eps=eps, seed=(seed+it))
            candidates.append((L, counts, coeff, m))

        # izberi elito
        candidates.sort(key=lambda x: x[0])
        elites = candidates[:elite]
        elite_loss = np.array([e[0] for e in elites])
        elite_coeff = np.stack([e[2] for e in elites], axis=0)

        # posodobi mu s tehtanim povprečjem (NES)
        w = np.exp(- (elite_loss - elite_loss.min()))
        w = w / (w.sum() + 1e-12)
        mu = (w[:,None] * elite_coeff).sum(axis=0)

        # rahlo zmanjšaj sigma, če smo dobri; sicer jo malo povečaj
        improve = elites[0][0] < best_score - 1e-6
        sigma *= 0.99 if improve else 1.01
        sigma = np.clip(sigma, 0.05*coef_sigma, 5*coef_sigma)

        # posodobi “najboljšega”
        if elites[0][0] < best_score:
            best_score, best_counts, best_mesh = elites[0][0], elites[0][1], elites[0][3].copy()

        if it % 5 == 0:
            tm.exchange.export.export_mesh(best_mesh, os.path.join(outdir, f"mesh_it{it:04d}.stl"))
        print(f"[{it:03d}] best L={best_score:.3f} (s,u,t)={best_counts}  mu_norm={np.linalg.norm(mu):.3f}")

        # zgodnja ustavitev
        if best_counts[0] == 1 and best_counts[1] == 1:
            print("Target (1 stable, 1 unstable) reached (by sampled test).")
            break

    tm.exchange.export.export_mesh(best_mesh, os.path.join(outdir, "final_candidate.stl"))
    return best_mesh

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("ML/ES tuning to (1 stable, 1 unstable) with exactly two mirrors")
    ap.add_argument("--stl", required=False,
                    default="/home/timen/Documents/Faks/Matematika-z-racunalnikom/ml_out2/final_candidate.stl",
                    help="Pot do STL datoteke")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--elite", type=int, default=10)
    ap.add_argument("--K", type=int, default=64, help="# kontrolnih smeri (RBF centri)")
    ap.add_argument("--sigma_deg", type=float, default=18.0, help="RBF širina (stopinje)")
    ap.add_argument("--step_coeff", type=float, default=1e-3, help="Skala deformacije po iteraciji")
    ap.add_argument("--coef_sigma", type=float, default=0.2, help="Inicialni std za koeficiente")
    ap.add_argument("--dirs", type=int, default=6000, help="# smeri na sferi za detekcijo")
    ap.add_argument("--eps", type=float, default=2e-3, help="majhen kotni zasuk za klasifikacijo")
    ap.add_argument("--out", type=str, default="ml_out3")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    mesh0 = load_mesh(args.stl, force_convex=True)
    best = optimize_shape(mesh0,
                          iters=args.iters, pop=args.pop, elite=args.elite,
                          K=args.K, sigma_deg=args.sigma_deg,
                          step_coeff=args.step_coeff, coef_sigma=args.coef_sigma,
                          n_dirs=args.dirs, eps=args.eps, seed=args.seed,
                          outdir=args.out)
    print("Saved:", os.path.join(args.out, "final_candidate.stl"))
