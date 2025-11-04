import os, math, json
import numpy as np
import trimesh as tm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from itertools import permutations, product

# --------------------------- Helpers ---------------------------
def to44(R3): T = np.eye(4); T[:3,:3] = R3; return T
def safe_unit(v, eps=1e-12): n=np.linalg.norm(v); return v/(n+eps)
def fibonacci_sphere(n_dirs, randomize=False, seed=0):
    rng = np.random.RandomState(seed) if randomize else None
    rnd = rng.rand()*n_dirs if rng is not None else 1.0
    pts=[]; off=2.0/n_dirs; inc=np.pi*(3.0-np.sqrt(5.0))
    for i in range(n_dirs):
        y=((i*off)-1)+(off/2); r=np.sqrt(max(0.0,1-y*y))
        phi=((i+rnd)%n_dirs)*inc; x=np.cos(phi)*r; z=np.sin(phi)*r
        pts.append(safe_unit(np.array([x,y,z],float)))
    return np.vstack(pts)

def principal_axes_R(mesh):
    I=mesh.moment_inertia; e,v=np.linalg.eigh(I); P=v[:,np.argsort(e)]
    if np.linalg.det(P)<0: P[:,2]*=-1; return P

# --------------------------- Load mesh ---------------------------
def load_mesh(path, force_convex=True):
    mesh=tm.load_mesh(path,force='mesh')
    if not isinstance(mesh,tm.Trimesh): mesh=mesh.dump().sum()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices(); mesh.remove_infinite_values()
    mesh.process(validate=True)
    if force_convex and not mesh.is_convex: mesh=mesh.convex_hull
    v=mesh.vertices-mesh.center_mass; r=np.sqrt((v**2).sum(1)).mean()
    if r>0: mesh.apply_scale(1.0/r); mesh.apply_translation(-mesh.center_mass)
    return mesh

def ensure_watertight_convex(mesh):
    if (not mesh.is_watertight) or (not mesh.is_convex):
        mesh=mesh.convex_hull
    return mesh

# --------------------------- Orientation ---------------------------
def reflect_points_across_plane(points,n):
    n=safe_unit(n); d=points@n; return points-2.0*d[:,None]*n[None,:]
def _mirror_score(V,normals):
    tree=cKDTree(V); sc=0.0
    for n in normals:
        Vref=reflect_points_across_plane(V,n)
        d,_=tree.query(Vref,k=1); sc+=float((d**2).mean())
    return sc
def auto_orient_to_two_mirrors(mesh,coarse_deg=5):
    P=principal_axes_R(mesh); V0=mesh.vertices.copy()
    normals=[np.array([1,0,0]),np.array([0,1,0])]
    best=(np.inf,np.eye(3))
    for perm in permutations([0,1,2]):
        B=P[:,perm]
        for signs in product([1.,-1.],[1.,-1.],[1.,-1.]):
            R0=B@np.diag(signs)
            if np.linalg.det(R0)<0: continue
            for deg in range(0,360,coarse_deg):
                Rz=R.from_euler('z',deg,degrees=True).as_matrix()
                Rtry=Rz@R0; V=(V0@Rtry.T)-V0.mean(0)
                sc=_mirror_score(V,normals)
                if sc<best[0]: best=(sc,Rtry)
    Rb=best[1]; m=mesh.copy(); m.apply_transform(to44(Rb))
    m.apply_translation(-m.center_mass); return m

# --------------------------- Equilibria ---------------------------
def oriented_mesh(mesh,up_dir):
    up_dir=safe_unit(up_dir); z=np.array([0,0,1])
    if np.allclose(up_dir,z): rot=R.identity()
    elif np.allclose(up_dir,-z): rot=R.from_rotvec(np.pi*np.array([1,0,0]))
    else:
        v=np.cross(up_dir,z); s=np.linalg.norm(v); c=float(np.dot(up_dir,z))
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]],float)
        Rm=np.eye(3)+vx+vx@vx*((1-c)/(s**2+1e-16)); rot=R.from_matrix(Rm)
    m2=mesh.copy(); m2.apply_transform(to44(rot.as_matrix())); return m2

def place_on_plane(mesh):
    zmin=float(mesh.vertices[:,2].min()); m2=mesh.copy()
    m2.apply_translation([0,0,-zmin]); return m2,float(m2.center_mass[2])

def com_height_for_updir(mesh,up_dir):
    mR=oriented_mesh(mesh,up_dir); _,h=place_on_plane(mR); return h

def classify_orientation(mesh, up_dir, eps=1e-3):
    h0 = com_height_for_updir(mesh, up_dir)

    def tilt(ax):
        ax = np.array(ax, dtype=float)        # 🔧 <-- DODANO
        Rt = R.from_rotvec(eps * ax)
        m = oriented_mesh(mesh, up_dir)
        m.apply_transform(to44(Rt.as_matrix()))
        _, h = place_on_plane(m)
        return h

    hx_p = tilt([ 1.0, 0.0, 0.0])
    hx_m = tilt([-1.0, 0.0, 0.0])
    hy_p = tilt([ 0.0, 1.0, 0.0])
    hy_m = tilt([ 0.0,-1.0, 0.0])

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


def _cluster(idx,h,dirs,pick='min'):
    if not idx: return []
    used=np.zeros(len(idx),bool); groups=[]
    for a in range(len(idx)):
        if used[a]: continue
        ia=idx[a]; pa=dirs[ia]; group=[ia]
        for b in range(a+1,len(idx)):
            if used[b]: continue
            ib=idx[b]; pb=dirs[ib]
            if float(np.dot(pa,pb))>0.995:
                used[b]=True; group.append(ib)
        used[a]=True; groups.append(group)
    rep=[]
    for g in groups:
        hs=h[g]; rep.append(g[int(np.argmin(hs))] if pick=='min' else g[int(np.argmax(hs))])
    return rep

def detect_equilibria(mesh,n_dirs=2000,eps=1e-3,seed=0):
    dirs=fibonacci_sphere(n_dirs,randomize=True,seed=seed)
    h=np.zeros(n_dirs)
    for i,d in enumerate(dirs): h[i]=com_height_for_updir(mesh,d)
    tree=cKDTree(dirs); nn=tree.query(dirs,k=13)[1][:,1:]
    labels=np.empty(n_dirs,object)
    for i in range(n_dirs):
        hi=h[i]; neigh=h[nn[i]]
        ismin=np.all(hi<=neigh-1e-8); ismax=np.all(hi>=neigh+1e-8)
        if ismin or ismax:
            lab,_=classify_orientation(mesh,dirs[i],eps=eps); labels[i]=lab
        else: labels[i]=None
    s=[i for i,l in enumerate(labels) if l=='stable']
    u=[i for i,l in enumerate(labels) if l=='unstable']
    t=[i for i,l in enumerate(labels) if l=='saddle']
    return {'dirs':dirs,'heights':h,
            'stable_idx':_cluster(s,h,dirs,'min'),
            'unstable_idx':_cluster(u,h,dirs,'max'),
            'saddle_idx':_cluster(t,h,dirs,'min')}

# --------------------------- Symmetry helpers ---------------------------
def vertex_normals(m): return m.vertex_normals.copy()
def reflect_points(V,n): n=safe_unit(n); d=V@n; return V-2*d[:,None]*n[None,:]
def enforce_mirror(m,n):
    n=safe_unit(n); V=m.vertices.copy(); Vm=reflect_points(V,n)
    tree=cKDTree(V); _,idx=tree.query(Vm,k=1); newV=V.copy()
    for i,j in enumerate(idx):
        vi=newV[i]; vj=newV[j]
        vi_new=0.5*(vi+reflect_points(vj[None,:],n)[0])
        newV[i]=vi_new; newV[j]=reflect_points(vi_new[None,:],n)[0]
    m.vertices=newV; 
    if hasattr(m,'_cache'): m._cache.clear(); return m

def break_plane_mirror(m,n,amount=1e-4):
    n=safe_unit(n); V=m.vertices.copy(); norms=m.vertex_normals
    sgn=np.sign(V@n)[:,None]; Vn=V+amount*sgn*norms
    new=m.copy(); new.vertices=Vn
    if hasattr(new,'_cache'): new._cache.clear()
    new=ensure_watertight_convex(new); new.apply_translation(-new.center_mass)
    return new

def ensure_exactly_two_mirrors(m,break_amount=1e-4):
    m=enforce_mirror(m,[1,0,0]); m=enforce_mirror(m,[0,1,0])
    m=break_plane_mirror(m,[0,0,1],break_amount)
    m=ensure_watertight_convex(m); m.apply_translation(-m.center_mass); return m

# --------------------------- Deformer ---------------------------
class SmoothDeformer:
    def __init__(self,mesh,K=32,sigma_deg=18.0,seed=0):
        self.K=K; self.centers=fibonacci_sphere(K,True,seed)
        self.sigma=np.deg2rad(sigma_deg)
        self.baseV=mesh.vertices.copy(); self.baseN=vertex_normals(mesh)
        self.com=mesh.center_mass.copy()
    def apply(self,mesh,coeff,scale=1.0):
        V=self.baseV.copy(); N=self.baseN.copy()
        dirV=(V-self.com); dirV=(dirV.T/(np.linalg.norm(dirV,1)+1e-12)).T
        total=np.zeros(len(V))
        for k in range(self.K):
            d=self.centers[k]; cosang=np.clip(dirV@d,-1,1)
            ang=np.arccos(cosang); w=np.exp(-0.5*(ang/(self.sigma+1e-12))**2)
            total+=coeff[k]*w
        disp=(scale*total)[:,None]*N; mesh.vertices=V+disp
        if hasattr(mesh,'_cache'): mesh._cache.clear(); return mesh

# --------------------------- Fitness ---------------------------
def fitness(m,n_dirs=2000,eps=1e-3,seed=0,w_s=3,w_u=2,w_t=0.2):
    eq=detect_equilibria(m,n_dirs,eps,seed)
    s,u,t=len(eq['stable_idx']),len(eq['unstable_idx']),len(eq['saddle_idx'])
    L=w_s*abs(s-1)+w_u*abs(u-1)+w_t*t; return L,(s,u,t)

# --------------------------- Optimization ---------------------------
from multiprocessing import Pool, cpu_count

def evaluate_candidate(args):
    coeff, m0, deform, step_coeff, n_dirs, eps, seed, it = args
    # vsak proces dobi svojo kopijo
    m = m0.copy()
    m = deform.apply(m, coeff, step_coeff)
    m = ensure_exactly_two_mirrors(m, 1e-4)
    L, c = fitness(m, n_dirs=n_dirs, eps=eps, seed=seed + it)
    return (L, c, coeff, m)

def optimize_shape(mesh_in, iters=2000, pop=96, elite=12, K=128,
                   sigma_deg=12, step_coeff=8e-4, coef_sigma=0.12,
                   n_dirs=6000, eps=2e-3, seed=0, outdir="ml_out4"):
    os.makedirs(outdir, exist_ok=True)
    def save_ckpt(it, best_score, best_counts):
        with open(os.path.join(outdir, "state.json"), "w") as f:
            json.dump({"iter": it, "best_score": best_score, "best_counts": best_counts}, f)

    # orientiraj in pripravi mrežo
    m0 = auto_orient_to_two_mirrors(mesh_in)
    m0 = ensure_exactly_two_mirrors(m0, 1e-4)
    tm.exchange.export.export_mesh(m0, os.path.join(outdir, "00_auto_oriented.stl"))

    deform = SmoothDeformer(m0, K, sigma_deg, seed)
    rng = np.random.RandomState(seed)
    mu = np.zeros(K)
    sigma = np.ones(K) * coef_sigma
    best_mesh = m0.copy()
    best_score, best_counts = fitness(best_mesh, n_dirs, eps, seed)
    print(f"[init] L={best_score:.3f} (s,u,t)={best_counts}")

    # --- paralelizacija ---
    num_cores = min(cpu_count(), 24)   # uporabi največ 24 jeder
    print(f"🧠 Using {num_cores} CPU cores for parallel evaluation.")

    for it in range(1, iters + 1):
        # pripravi argumente za vse kandidate
        args_list = [
            (mu + sigma * rng.randn(K), m0, deform, step_coeff, n_dirs, eps, seed, it)
            for _ in range(pop)
        ]

        # paralelno izračunaj populacijo
        with Pool(processes=num_cores) as pool:
            cand = pool.map(evaluate_candidate, args_list)

        cand.sort(key=lambda x: x[0])
        elites = cand[:elite]
        eL = np.array([e[0] for e in elites])
        eC = np.stack([e[2] for e in elites])

        w = np.exp(-(eL - eL.min()))
        w /= w.sum() + 1e-12
        mu = (w[:, None] * eC).sum(axis=0)

        improve = elites[0][0] < best_score - 1e-6
        sigma *= 0.99 if improve else 1.01
        sigma = np.clip(sigma, 0.05 * coef_sigma, 5 * coef_sigma)

        if elites[0][0] < best_score:
            best_score, best_counts, best_mesh = elites[0][0], elites[0][1], elites[0][3].copy()

        # annealing
        if it % 200 == 0:
            step_coeff = max(step_coeff * 0.9, 2e-4)
        if it in [int(0.25 * iters), int(0.5 * iters), int(0.75 * iters)]:
            n_dirs = int(n_dirs * 1.5)

        if it % 10 == 0:
            save_ckpt(it, best_score, best_counts)
            tm.exchange.export.export_mesh(best_mesh, os.path.join(outdir, f"mesh_it{it:04d}.stl"))

        print(f"[{it:04d}] L={best_score:.3f} (s,u,t)={best_counts} step={step_coeff:.1e}")

        if best_counts == (1, 1):
            print("✅ Target (1,1) reached!")
            break

    tm.exchange.export.export_mesh(best_mesh, os.path.join(outdir, "final_candidate.stl"))
    return best_mesh


# --------------------------- CLI ---------------------------
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser("Precise Gömböc optimizer (2 mirrors)")
    ap.add_argument("--stl",default="/home/timen/Documents/Faks/Matematika-z-racunalnikom/pristop2/ml_out4/mesh_it0630.stl")
    ap.add_argument("--iters",type=int,default=20)
    ap.add_argument("--pop",type=int,default=96)
    ap.add_argument("--elite",type=int,default=12)
    ap.add_argument("--K",type=int,default=128)
    ap.add_argument("--sigma_deg",type=float,default=12.0)
    ap.add_argument("--step_coeff",type=float,default=8e-4)
    ap.add_argument("--coef_sigma",type=float,default=0.12)
    ap.add_argument("--dirs",type=int,default=6000)
    ap.add_argument("--eps",type=float,default=2e-3)
    ap.add_argument("--out",type=str,default="ml_out5")
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    mesh0=load_mesh(a.stl,True)
    best=optimize_shape(mesh0,a.iters,a.pop,a.elite,a.K,a.sigma_deg,
                        a.step_coeff,a.coef_sigma,a.dirs,a.eps,a.seed,a.out)
    print("Saved:",os.path.join(a.out,"final_candidate.stl"))
