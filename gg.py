"""
gomboc_search_high_quality_mt.py

Infinite Gömböc search using multi-threading.
Prints candidates and saves high-quality STL for 3D printing.
"""

import numpy as np
from scipy.special import sph_harm
from scipy.spatial import cKDTree, Delaunay
from stl import mesh
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# Utilities (same as before)
# ---------------------------
def fibonacci_sphere(samples=1000):
    N = samples
    points = np.zeros((N, 3))
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(N):
        y = 1 - (i / float(N - 1)) * 2
        radius = np.sqrt(1 - y*y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points[i] = np.array([x, y, z])
    return points

def eval_support_function(coeffs, u):
    x, y, z = u[...,0], u[...,1], u[...,2]
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    theta = np.arctan2(y, x) % (2*np.pi)
    h = np.zeros_like(phi, dtype=float)
    for (ell, m), a in coeffs.items():
        Y_lm = sph_harm(m, ell, theta, phi)
        h += np.real(a * Y_lm)
    return h

def support_to_surface_points(coeffs, normals):
    h = eval_support_function(coeffs, normals)
    N = len(normals)
    points = np.zeros((N,3))
    tree = cKDTree(normals)
    for i in range(N):
        u = normals[i]
        h0 = h[i]
        dists, idxs = tree.query(u, k=20)
        neighbor_idxs = idxs if np.ndim(idxs) > 0 else [idxs]
        A = []
        b = []
        for j in neighbor_idxs:
            if j == i: continue
            v = normals[j]
            dh = h[j]-h0
            dv = v-u
            A.append(dv)
            b.append(dh)
        if len(A) < 3:
            points[i] = h0*u
            continue
        A = np.array(A)
        b = np.array(b)
        g, *_ = np.linalg.lstsq(A,b,rcond=None)
        g_tangent = g - np.dot(g,u)*u
        points[i] = g_tangent + h0*u
    return points

def check_convexity(coeffs, normals, points):
    N = len(normals)
    tree = cKDTree(points)
    for i in range(N):
        p = points[i]
        u = normals[i]
        dists, idxs = tree.query(p,k=12)
        neigh = points[idxs]
        signed = np.dot(neigh-p,u)
        if np.any(signed < -1e-6):
            return False
    return True

def classify_equilibria(points, normals, gravity_dirs, neighbor_k=12):
    N = len(points)
    eq_indices=[]
    eq_types=[]
    tree = cKDTree(points)
    for v in gravity_dirs:
        s = points @ v
        dists, idxs = tree.query(points, k=neighbor_k)
        for i in range(N):
            neigh_idx = idxs[i]
            neigh_s = s[neigh_idx]
            si = s[i]
            if np.all(neigh_s>=si-1e-12) and np.any(neigh_s>si+1e-12):
                eq_indices.append(i)
                eq_types.append('min')
            elif np.all(neigh_s<=si+1e-12) and np.any(neigh_s<si-1e-12):
                eq_indices.append(i)
                eq_types.append('max')
            else:
                if np.any(neigh_s>si+1e-12) and np.any(neigh_s<si-1e-12):
                    eq_indices.append(i)
                    eq_types.append('saddle')
    if len(eq_indices)==0: return {'S':0,'H':0,'U':0}
    coords = points[eq_indices]
    thresh=1e-3
    labels=-np.ones(len(coords),dtype=int)
    cur_label=0
    from collections import Counter
    for i in range(len(coords)):
        if labels[i]!=-1: continue
        stack=[i]
        labels[i]=cur_label
        while stack:
            a = stack.pop()
            for b in range(len(coords)):
                if labels[b]!=-1: continue
                if np.linalg.norm(coords[a]-coords[b])<thresh:
                    labels[b]=cur_label
                    stack.append(b)
        cur_label+=1
    clusters_info={}
    for idx_label, typ in zip(labels, eq_types):
        clusters_info.setdefault(idx_label,[]).append(typ)
    S=H=U=0
    for typs in clusters_info.values():
        main=Counter(typs).most_common(1)[0][0]
        if main=='min': S+=1
        elif main=='max': U+=1
        else: H+=1
    return {'S':S,'H':H,'U':U}

def make_coeffs_from_vector(vec):
    basis=[(0,0),(1,-1),(1,0),(1,1),(2,-2),(2,-1),(2,0),(2,1),(2,2)]
    return { (ell,m):v for (ell,m),v in zip(basis,vec)}

def test_candidate(vec, sample_normals=1200, gravity_samples=42):
    coeffs = make_coeffs_from_vector(vec)
    normals = fibonacci_sphere(sample_normals)
    if (0,0) not in coeffs: coeffs[(0,0)]=0.3
    else: coeffs[(0,0)]+=0.3
    points = support_to_surface_points(coeffs, normals)
    convex = check_convexity(coeffs, normals, points)
    if not convex:
        return {'convex':False,'S':-1,'H':-1,'U':-1,'points':points,'normals':normals,'coeffs':coeffs}
    grav_dirs = fibonacci_sphere(gravity_samples)
    res = classify_equilibria(points, normals, grav_dirs)
    return {'convex':convex,'S':res['S'],'H':res['H'],'U':res['U'],'points':points,'normals':normals,'coeffs':coeffs}

def save_high_quality_stl(points, filename):
    x,y,z=points[:,0],points[:,1],points[:,2]
    r=np.linalg.norm(points,axis=1)
    phi=np.arccos(np.clip(z/r,-1.0,1.0))
    theta=np.arctan2(y,x)
    pts2d=np.column_stack([theta,phi])
    tri=Delaunay(pts2d)
    faces=tri.simplices
    m=mesh.Mesh(np.zeros(faces.shape[0],dtype=mesh.Mesh.dtype))
    for i,f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j]=points[f[j]]
    m.save(filename)

# ---------------------------
# Parallel infinite search
# ---------------------------
def infinite_search_mt(num_threads=4):
    os.makedirs("candidates", exist_ok=True)
    found_count = 0
    base = np.zeros(9)
    base[0]=0.3

    def worker():
        vec = base.copy()
        vec[2]=np.random.uniform(-0.06,0.06)
        vec[6]=np.random.uniform(-0.06,0.06)
        return vec, test_candidate(vec)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker):None for _ in range(num_threads)}
        while True:
            for fut in as_completed(futures):
                vec, out = fut.result()
                # submit new task immediately
                futures[executor.submit(worker)] = None
                if out['convex'] and out['S']==1 and out['H']==0 and out['U']==1:
                    found_count+=1
                    print(f"Candidate #{found_count}: vec[2]={vec[2]:.6f}, vec[6]={vec[6]:.6f}")
                    pts=out['points']
                    mins, maxs=pts.min(axis=0), pts.max(axis=0)
                    dims=maxs-mins
                    print(f"Bounding box: x={dims[0]:.3f}, y={dims[1]:.3f}, z={dims[2]:.3f}")
                    filename=f"candidates/gomboc_{found_count}.stl"
                    save_high_quality_stl(pts,filename)
                    print(f"Saved STL: {filename}\n")

# ---------------------------
# Run
# ---------------------------
if __name__=="__main__":
    print("Running multi-threaded Gömböc search...")
    infinite_search_mt(num_threads=8)  # adjust threads to CPU cores