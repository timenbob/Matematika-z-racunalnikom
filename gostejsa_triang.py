import trimesh as tm
from trimesh import remesh

mesh = tm.load_mesh("/home/timen/Documents/Faks/Matematika-z-racunalnikom/gombocid2/best.stl", force='mesh')

# variant A1: ciljna največja dolžina roba (manjša -> gostejša)
max_edge = 0.5  # prilagodi: manjše število => bolj fina mreža
# remesh zahteva posebej oglišča in ploskve
vertices, faces = remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge)

# ustvari novo mrežo
mesh_dense = tm.Trimesh(vertices=vertices, faces=faces, process=True)

mesh_dense.apply_translation(-mesh_dense.center_mass)  # če rabiš centrirano
mesh_dense.export("novi3.stl")
print("Saved input_dense.stl, triangles:", len(mesh_dense.faces))
