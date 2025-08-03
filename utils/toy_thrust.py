from sklearn.decomposition import PCA
import numpy as np

def calculate_thrust_axis_pca(px_ls, py_ls, pz_ls):
    # Stack components into a (N, 3) array
    particles_momenta = np.vstack((px_ls, py_ls, pz_ls)).T  # shape: (N_particles, 3)
    
    # Perform PCA
    pca = PCA(n_components=1)
    pca.fit(particles_momenta)
    
    # The thrust axis is the direction of the first principal component
    thrust_axis = pca.components_[0]
    
    return thrust_axis


px_ls = [1.0, -0.4, -0.6]
py_ls = [-0.8, 1.2, -0.4]
pz_ls = [-0.3, -0.7, 1.0]

thrust_axis = calculate_thrust_axis_pca(px_ls, py_ls, pz_ls)
print("Thrust axis (PCA approximation):", thrust_axis)



def compute_momentum_tensor(particles_px, particles_py, particles_pz):
    # Create momentum components array
    p = np.vstack([particles_px, particles_py, particles_pz]).T  # Shape (n_particles, 3)
    
    # Normalize the momenta
    total_p2 = np.sum(np.linalg.norm(p, axis=1)**2)  # Sum of p^2 for each particle
    p2 = np.linalg.norm(p, axis=1)**2  # p^2 for each particle
    
    # Construct the momentum tensor M_ab
    M_ab = np.zeros((3, 3))  # 3x3 matrix
    
    for i in range(len(particles_px)):
        # Outer product of momentum components (p_a * p_b)
        M_ab += np.outer(p[i], p[i])  # Outer product (p_a * p_b)
    
    M_ab /= total_p2  # Normalize by sum(p^2)
    
    return M_ab

def compute_eigenvalues(M_ab):
    # Diagonalize the momentum tensor to find eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(M_ab)  # eigenvalues are sorted in ascending order
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    return eigenvalues

def compute_sphericity(eigenvalues):
    # Compute sphericity
    lambda1, lambda2, lambda3 = eigenvalues
    print(lambda1)
    print(lambda2)
    print(lambda3)
    sphericity = (3/2) * ( lambda2 + lambda3 )
    return sphericity

def compute_aplanarity(eigenvalues):
    # Compute aplanarity
    lambda1, lambda2, lambda3 = eigenvalues
    aplanarity = (3/2) * lambda3
    return aplanarity

# Example usage
particles_px = np.array([10, 2, -1, 4])
particles_py = np.array([5, -3, 1, 6])
particles_pz = np.array([1, 3, -2, 0])

# Compute momentum tensor
M_ab = compute_momentum_tensor(particles_px, particles_py, particles_pz)

# Compute eigenvalues
eigenvalues = compute_eigenvalues(M_ab)

# Compute sphericity and aplanarity
sphericity = compute_sphericity(eigenvalues)
aplanarity = compute_aplanarity(eigenvalues)

print(f"Sphericity: {sphericity}")
print(f"Aplanarity: {aplanarity}")
