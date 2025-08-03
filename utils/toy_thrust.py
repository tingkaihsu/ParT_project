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


px_ls = [1.0, 0.4, -0.3]
py_ls = [0.5, 1.2, -0.4]
pz_ls = [0.3, 0.7, 1.0]

thrust_axis = calculate_thrust_axis_pca(px_ls, py_ls, pz_ls)
print("Thrust axis (PCA approximation):", thrust_axis)

