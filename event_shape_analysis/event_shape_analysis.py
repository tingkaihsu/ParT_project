import numpy as np
from sklearn.decomposition import PCA
# import math
# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def get_final_states(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        # Vertex ID record the what vertex is currently
        parent_vertex_id = 0
        B_plus_final_states_arr = []
        B_minus_final_states_arr = []
        
        
        for line in lines:
            # New event
            if line.startswith('E'):
                if parent_vertex_id == 0:
                    # First line event
                    pass
                else:
                    B_plus_final_states_arr.append( B_plus_final_states )
                    B_minus_final_states_arr.append( B_minus_final_states)

                parent_vertex_id = 0
                B_plus_final_states = []
                B_minus_final_states = []
                
                B_plus_vertices = []
                B_minus_vertices = []
            # New vertex
            elif line.startswith('V'):
                parent_vertex_id = parent_vertex_id + 1
            # New particle
            elif line.startswith('P'):
                # Split the line 
                partcl_dt = line.split()
                cur_partcl = (
                        int(partcl_dt[2]),                # Particle ID (pdgid)
                        np.float64(partcl_dt[3]),         # Momentum in x (px)
                        np.float64(partcl_dt[4]),         # Momentum in y (py)
                        np.float64(partcl_dt[5]),         # Momentum in z (pz)
                        np.float64(partcl_dt[6]),         # Energy
                        np.float64(partcl_dt[7]),         # Mass
                        abs(int(partcl_dt[11])),          # Child vertex
                        parent_vertex_id                  # Current vertex ID
                )
                # Only record particles whose parent_vertex_id from B+ meson or B- meson
                # and its child vertex = 0
                child_vertex_id = abs(int(partcl_dt[11]))
                pdgid = int(partcl_dt[2])
                # First check whether it is B+ meson and B- meson
                if pdgid == 521:
                    B_plus_vertices.append(child_vertex_id)
                elif pdgid == -521:
                    B_minus_vertices.append(child_vertex_id)
                else:
                    pass
                
                # If parent vertex from B+ meson decay, record the child vertex in B+ meson decay
                # If null child vertex, then it is the final states
                for i in range(len(B_plus_vertices)):
                    if parent_vertex_id == B_plus_vertices[i] and not child_vertex_id == 0:
                        B_plus_vertices.append( child_vertex_id )
                    elif parent_vertex_id == B_plus_vertices[i] and child_vertex_id == 0:
                        B_plus_final_states.append(cur_partcl)
                    else:
                        pass
                        
                for i in range(len(B_minus_vertices)):
                    if parent_vertex_id == B_minus_vertices[i] and not child_vertex_id == 0:
                        B_minus_vertices.append( child_vertex_id )
                    elif parent_vertex_id == B_minus_vertices[i] and child_vertex_id == 0:
                        B_minus_final_states.append(cur_partcl)
                    else:
                        pass
        
        B_plus_final_states_arr.append( B_plus_final_states )
        B_minus_final_states_arr.append( B_minus_final_states)
        parent_vertex_id = 0
        B_plus_final_states = []
        B_minus_final_states = []
        
        B_plus_vertices = []
        B_minus_vertices = []

        return B_plus_final_states_arr, B_minus_final_states_arr
    
def get_thrust_axis(px_s, py_s, pz_s):
    if len(px_s) == 0 or len(py_s) == 0 or len(pz_s) == 0:
        raise ValueError("Empty momentum components passed to get_thrust_axis.")

    particles_momenta = np.vstack((px_s, py_s, pz_s)).T
    pca = PCA(n_components=1)
    pca.fit(particles_momenta)
    return pca.components_[0]

def get_thrust(px, py, pz, thrust_axis):
    total_momentum_magnitude = 0
    projected_momentum_sum = 0

    for i in range(len(px)):
        momentum_vec = np.array([px[i], py[i], pz[i]])
        momentum_mag = np.linalg.norm(momentum_vec)
        projection = np.abs(np.dot(momentum_vec, thrust_axis))
        
        total_momentum_magnitude += momentum_mag
        projected_momentum_sum += projection

    if total_momentum_magnitude == 0:
        return 0.0  # avoid division by zero

    thrust = projected_momentum_sum / total_momentum_magnitude
    return thrust

def get_Q(particles_px, particles_py, particles_pz):
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

    eigenvalues, eigenvectors = np.linalg.eigh(M_ab)  # eigenvalues are sorted in ascending order
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort eigenvalues in descending order

    return eigenvalues

def get_sphericity(eigenvalues):
    # Compute sphericity
    lambda1, lambda2, lambda3 = eigenvalues
    sphericity = (3/2) * ( lambda2 + lambda3 )
    return sphericity

def get_aplanarity(eigenvalues):
    # Compute aplanarity
    lambda1, lambda2, lambda3 = eigenvalues
    aplanarity = (3/2) * lambda3
    return aplanarity


B_plus_arr, B_minus_arr = get_final_states("hepMCtest")

B_plus_final_state_px_arr = []
B_plus_final_state_py_arr = []
B_plus_final_state_pz_arr = []

B_plus_final_state_energy_arr = []
B_plus_final_state_mass_arr = []

for B_meson in B_plus_arr:
    px_s = []
    py_s = []
    pz_s = []
    energy_s = []
    mass_s = []
    for final_states in B_meson:
        final_state_px = final_states[1]
        final_state_py = final_states[2]
        final_state_pz = final_states[3]
        final_state_energy = final_states[4]
        final_state_mass = final_states[5]

        px_s.append(final_state_px)
        py_s.append(final_state_py)
        pz_s.append(final_state_pz)
        energy_s.append(final_state_energy)
        mass_s.append(final_state_mass)
    B_plus_final_state_px_arr.append(px_s)
    B_plus_final_state_py_arr.append(py_s)
    B_plus_final_state_pz_arr.append(pz_s)
    B_plus_final_state_energy_arr.append(energy_s)
    B_plus_final_state_mass_arr.append(mass_s)


B_minus_final_state_px_arr = []
B_minus_final_state_py_arr = []
B_minus_final_state_pz_arr = []

B_minus_final_state_energy_arr = []
B_minus_final_state_mass_arr = []

for B_meson in B_minus_arr:
    px_s = []
    py_s = []
    pz_s = []
    energy_s = []
    mass_s = []
    for final_states in B_meson:
        final_state_px = final_states[1]
        final_state_py = final_states[2]
        final_state_pz = final_states[3]
        final_state_energy = final_states[4]
        final_state_mass = final_states[5]

        px_s.append(final_state_px)
        py_s.append(final_state_py)
        pz_s.append(final_state_pz)
        energy_s.append(final_state_energy)
        mass_s.append(final_state_mass)
    B_minus_final_state_px_arr.append(px_s)
    B_minus_final_state_py_arr.append(py_s)
    B_minus_final_state_pz_arr.append(pz_s)
    B_minus_final_state_energy_arr.append(energy_s)
    B_minus_final_state_mass_arr.append(mass_s)

# Sum up all final states first
total_arr = []
for B_plus, B_minus in zip( B_plus_arr, B_minus_arr ):
    total = B_plus + B_minus
    total_arr.append(total)

total_final_state_px_arr = []
total_final_state_py_arr = []
total_final_state_pz_arr = []

total_final_state_energy_arr = []
total_final_state_mass_arr = []

for B_meson in total_arr:
    px_s = []
    py_s = []
    pz_s = []
    energy_s = []
    mass_s = []
    for final_states in B_meson:
        final_state_px = final_states[1]
        final_state_py = final_states[2]
        final_state_pz = final_states[3]
        final_state_energy = final_states[4]
        final_state_mass = final_states[5]

        px_s.append(final_state_px)
        py_s.append(final_state_py)
        pz_s.append(final_state_pz)
        energy_s.append(final_state_energy)
        mass_s.append(final_state_mass)
    total_final_state_px_arr.append(px_s)
    total_final_state_py_arr.append(py_s)
    total_final_state_pz_arr.append(pz_s)
    total_final_state_energy_arr.append(energy_s)
    total_final_state_mass_arr.append(mass_s)


thrust_arr = []
sphericity_arr = []
aplanarity_arr = []

miss = 0
for i in range(len(total_final_state_px_arr)):
    px = total_final_state_px_arr[i]
    py = total_final_state_py_arr[i]
    pz = total_final_state_pz_arr[i]

    if len(px) == 0 or len(py) == 0 or len(pz) == 0:
        # print(f"Skipping empty event at index {i}")
        miss = miss + 1
        continue

    thrust_axis = get_thrust_axis(px, py, pz)
    thrust = get_thrust(px, py, pz, thrust_axis)
    # Calculate the momentum tensor and its eigenvalues
    eigenvalues = get_Q(px, py, pz)

    # Calculate sphericity and aplanarity
    sphericity = get_sphericity(eigenvalues)
    aplanarity = get_aplanarity(eigenvalues)

    # Append the computed values to lists
    thrust_arr.append(thrust)
    sphericity_arr.append(sphericity)
    aplanarity_arr.append(aplanarity)

    # print(f"Event {i}: Thrust = {thrust_value:.4f}, Sphericity = {sphericity:.4f}, Aplanarity = {aplanarity:.4f}")

# Plotting using Matplotlib
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot Thrust Distribution
plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
plt.hist(thrust_arr, bins=50, range=(0, 1), color='blue', alpha=0.7)
plt.title("Thrust Distribution")
plt.xlabel("Thrust")
plt.ylabel("Frequency")

# Plot Sphericity Distribution
plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
plt.hist(sphericity_arr, bins=50, range=(0, 1), color='green', alpha=0.7)
plt.title("Sphericity Distribution")
plt.xlabel("Sphericity")
plt.ylabel("Frequency")

# Plot Aplanarity Distribution
plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
plt.hist(aplanarity_arr, bins=50, range=(0, 1), color='red', alpha=0.7)
plt.title("Aplanarity Distribution")
plt.xlabel("Aplanarity")
plt.ylabel("Frequency")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

print(miss)