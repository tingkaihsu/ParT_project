import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import logging
from sklearn.decomposition import PCA

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

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

def get_final_states(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        # Vertex ID record the what vertex is currently

        parent_vertex_id = 0
        B_plus_final_states_arr = []
        B_minus_final_states_arr = []
        other_final_states_arr = []
        
        for line in lines:
            # New event
            if line.startswith('E'):
                if parent_vertex_id == 0:
                    # First line event
                    pass
                else:
                    B_plus_final_states_arr.append( B_plus_final_states )
                    B_minus_final_states_arr.append( B_minus_final_states)
                    other_final_states_arr.append( other_final_states )

                parent_vertex_id = 0
                B_plus_final_states = []
                B_minus_final_states = []
                other_final_states = []
                
                B_plus_vertices = []
                B_minus_vertices = []
                other_vertices = []
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
                elif not pdgid == 300553 and not pdgid == 521 and not pdgid == -521 and parent_vertex_id == 1:
                    # Sometimes there is a photon
                    if child_vertex_id == 0:
                        other_final_states.append( cur_partcl )
                    else:
                        other_vertices.append(child_vertex_id)
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
                
                for i in range( len(other_vertices) ):
                    if parent_vertex_id == other_vertices[i] and not child_vertex_id == 0:
                        other_vertices.append( child_vertex_id )
                    elif parent_vertex_id == other_vertices[i] and child_vertex_id == 0:
                        other_final_states.append( cur_partcl )
                    else:
                        pass

        # Final event
        B_plus_final_states_arr.append( B_plus_final_states )
        B_minus_final_states_arr.append( B_minus_final_states)
        other_final_states_arr.append( other_final_states )

        parent_vertex_id = 0
        B_plus_final_states = []
        B_minus_final_states = []
        other_final_states = []
        
        B_plus_vertices = []
        B_minus_vertices = []
        other_vertices = []

        return B_plus_final_states_arr, B_minus_final_states_arr, other_final_states_arr
def SetAKArr(filepath):
    B_plus_arr, B_minus_arr, other_arr = get_final_states( filepath )

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

    other_final_state_px_arr = []
    other_final_state_py_arr = []
    other_final_state_pz_arr = []

    other_final_state_energy_arr = []
    other_final_state_mass_arr = []

    for others in other_arr:
        px_s = []
        py_s = []
        pz_s = []
        energy_s = []
        mass_s = []
        for final_states in others:
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

        other_final_state_px_arr.append(px_s)
        other_final_state_py_arr.append(py_s)
        other_final_state_pz_arr.append(pz_s)
        other_final_state_energy_arr.append(energy_s)
        other_final_state_mass_arr.append(mass_s)

    _label = []
    
    total_arr = []

    for B_plus, B_minus, others in zip(B_plus_arr, B_minus_arr, other_arr):
        if len(B_plus) == 0 and len(B_minus) == 0:
            total = others
            _label.append( False )
        else:
            total = B_plus + B_minus
            _label.append( True )
        total_arr.append( total )

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

    v = {
        'part_px': total_final_state_px_arr,
        'part_py': total_final_state_py_arr,
        'part_pz': total_final_state_pz_arr,
        'part_energy': total_final_state_energy_arr,
        'part_mass': total_final_state_mass_arr,
        'label': np.stack(_label, axis=-1)  # Shape: (num_experiments,)
    }
    return v

# SetAKArr( "../event_shape_analysis/hepMCtest" )


def readFile(data_in_filepath, parquet_out_filepath):
    # You need to decide which fields to include; here I included part_charge and part_pdg as well
    schema = pa.schema([
        pa.field('label', pa.float64(), nullable=False),
        pa.field('part_px', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_py', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_pz', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_energy', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_mass', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False)
    ])

    data = SetAKArr(data_in_filepath)

    # Create a DataFrame, converting numpy arrays to lists for pandas compatibility
    df = pd.DataFrame({
        key: [arr.tolist() if isinstance(arr, (np.ndarray)) else arr for arr in val]  # Handle list or ndarray
        for key, val in data.items()
    })

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, parquet_out_filepath)
    logging.info(f"Written parquet file: {parquet_out_filepath}")


# Example usage
# readFile('../raw_data/train.txt', '../data/Bmeson/train_file.parquet')
readFile('../event_shape_analysis/hepMCtest', '../data/Bmeson/testing_file.parquet')
# readFile('../raw_data/testing.txt', '../data/Bmeson/testing_file.parquet')