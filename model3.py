import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import trimesh
from stl import mesh

def gomboc_parametric(u, v, a=1.0, b=0.8, c=0.6, epsilon=0.1):
    """
    Generate Gömböc-like shape using parametric equations
    Based on modified ellipsoid with carefully tuned parameters
    """
    # Modified ellipsoid with perturbation to create mono-monostatic property
    x = a * np.sin(u) * np.cos(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(u)
    
    # Add perturbation to break symmetry and create single stable point
    perturbation = epsilon * np.sin(2*u) * np.cos(3*v)
    z += perturbation
    
    return x, y, z

def generate_gomboc_points(resolution=100):
    """Generate point cloud for Gömböc"""
    u = np.linspace(0, np.pi, resolution)  # theta
    v = np.linspace(0, 2 * np.pi, resolution)  # phi
    
    U, V = np.meshgrid(u, v)
    
    # Experiment with these parameters to approach true Gömböc properties
    points = np.array(gomboc_parametric(U, V, a=1.0, b=0.95, c=0.7, epsilon=0.05))
    
    # Reshape to Nx3 array
    return points.reshape(3, -1).T

def create_convex_mesh(points):
    """Create convex mesh from points using Convex Hull"""
    hull = ConvexHull(points)
    
    # Create mesh
    vertices = points
    faces = hull.simplices
    
    return vertices, faces

def save_as_stl(vertices, faces, filename="gomboc.stl"):
    """Save mesh as STL file"""
    # Create the mesh
    gomboc_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, face in enumerate(faces):
        for j in range(3):
            gomboc_mesh.vectors[i][j] = vertices[face[j], :]
    
    # Save to file
    gomboc_mesh.save(filename)
    print(f"Gömböc saved as {filename}")

def plot_gomboc(vertices, faces):
    """3D visualization of the Gömböc"""
    fig = plt.figure(figsize=(12, 10))
    
    # Plot 1: 3D surface
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                    triangles=faces, alpha=0.8, cmap='viridis')
    ax1.set_title('3D Gömböc')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Different viewpoints
    views = [(0, 0), (45, 45), (90, 0), (45, -45)]
    titles = ['Front', 'Isometric', 'Side', 'Back']
    
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 4, i+5, projection='3d')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       triangles=faces, alpha=0.8, cmap='viridis')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_equilibrium(vertices):
    """Basic analysis of equilibrium points"""
    # Calculate center of mass (assuming uniform density)
    com = np.mean(vertices, axis=0)
    
    # Find potential equilibrium points (local minima/maxima in height)
    heights = vertices[:, 2]
    min_height_idx = np.argmin(heights)
    max_height_idx = np.argmax(heights)
    
    print(f"Center of mass: {com}")
    print(f"Lowest point: {vertices[min_height_idx]}")
    print(f"Highest point: {vertices[max_height_idx]}")
    
    return min_height_idx, max_height_idx

# Main execution
if __name__ == "__main__":
    print("Generating Gömböc...")
    
    # Generate points
    points = generate_gomboc_points(resolution=80)
    
    # Create convex mesh
    vertices, faces = create_convex_mesh(points)
    
    # Analyze equilibrium
    min_idx, max_idx = analyze_equilibrium(vertices)
    
    # Visualize
    plot_gomboc(vertices, faces)
    
    # Save as STL for 3D printing
    save_as_stl(vertices, faces, "gomboc_model.stl")
    
    print("Gömböc generation complete!")
    print("\nNote: This is an approximation. For a true Gömböc:")
    print("1. Adjust parameters in gomboc_parametric()")
    print("2. Consider using higher resolution")
    print("3. Verify equilibrium points experimentally")