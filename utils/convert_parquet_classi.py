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
    """
    Process the raw input file and compute the thrust for each experiment.
    
    Parameters:
    - filepath: Path to the input text file containing particle data.
    
    Returns:
    - A dictionary with processed data (particle momenta, labels, etc.).
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Initialize lists for storing data
    px, py, pz, energy, mass, charge, pdg, thrust, jetiness, sphericality = [], [], [], [], [], [], [], [], [], []
    px_ls, py_ls, pz_ls, energy_ls, mass_ls, charge_ls, pdg_ls = [], [], [], [], [], [], []
    _label1, _label2, _label3, _label4, _label5 = [], [], [], [], []

    n = 0
    for line in lines:
        if line.startswith('E'):  # Event header line
            if n != 0:
                # Store the particle data for the current event
                px.append(np.array(px_ls, dtype=float))
                py.append(np.array(py_ls, dtype=float))
                pz.append(np.array(pz_ls, dtype=float))
                energy.append(np.array(energy_ls, dtype=float))
                mass.append(np.array(mass_ls, dtype=float))
                charge.append(np.array(charge_ls, dtype=int))
                pdg.append(np.array(pdg_ls, dtype=int))


                # Check for photons (pdg code 22)
                for p in pdg_ls:
                    if p == 22:
                        print("Warning: Photon detected!")
                thrust.append(thrust_value)
                sphericality.append(sph_value)
                jetiness.append(jetiness_value)
                # Clear the per-experiment lists
                px_ls, py_ls, pz_ls, energy_ls, mass_ls, charge_ls, pdg_ls = [], [], [], [], [], [], []

            # Parse experiment labels from the header line
            exp_inf = line.split()
            _label1.append(float(exp_inf[1]))
            _label2.append(float(exp_inf[2]))
            _label3.append(float(exp_inf[3]))
            _label4.append(float(exp_inf[4]))
            _label5.append(float(exp_inf[5]))

            n = 0  # Reset particle count for the next experiment
        else:  # Particle data line
            par = line.split()
            if float(par[1]) == 22:  # Ignore photons (pdg == 22)
                continue
            else:
                n += 1
                # Append particle data to respective lists
                charge_ls.append(int(par[0]))
                pdg_ls.append(int(par[1]))
                px_ls.append(float(par[2]))
                py_ls.append(float(par[3]))
                pz_ls.append(float(par[4]))
                energy_ls.append(float(par[5]))
                mass_ls.append(float(par[6]))

    # Append the last experiment data (after file ends)
    if n > 0:
        px.append(np.array(px_ls, dtype=float))
        py.append(np.array(py_ls, dtype=float))
        pz.append(np.array(pz_ls, dtype=float))
        energy.append(np.array(energy_ls, dtype=float))
        mass.append(np.array(mass_ls, dtype=float))
        charge.append(np.array(charge_ls, dtype=int))
        pdg.append(np.array(pdg_ls, dtype=int))

        # Calculate thrust for the current event
        thrust_value = get_thrust(px_ls, py_ls, pz_ls)
        sph_value = get_sphericality(px_ls,py_ls,pz_ls)
        jetiness_value = get_jetiness(px_ls,py_ls,pz_ls,energy_ls)
        # if sph_value == 0:
        #     print(f"Thrust Value: {thrust_value}")
        #     print(f"Sphericality: {sph_value}")
        #     print(f"Jetiness: {jetiness_value}")
        thrust.append(thrust_value)
        sphericality.append(sph_value)
        jetiness.append(jetiness_value)
    # Construct the final dictionary for output
    v = {
        'part_px': px,
        'part_py': py,
        'part_pz': pz,
        'part_energy': energy,
        'part_mass': mass,
        'part_charge': charge,
        'part_pdg': pdg,
        'thrust': thrust,
        'sphericality': sphericality,
        'jetiness': jetiness,
        'label': np.stack(_label5, axis=-1)  # Shape: (num_experiments,)
    }

    # Check for NaNs in the particle data (optional but recommended)
    for arr_list in [px, py, pz, energy, mass]:
        for arr in arr_list:
            if np.isnan(arr).any():
                logging.warning('NaN detected in data arrays!')

    return v


def readFile(data_in_filepath, parquet_out_filepath):
    # You need to decide which fields to include; here I included part_charge and part_pdg as well
    schema = pa.schema([
        pa.field('label', pa.float64(), nullable=False),
        pa.field('jetiness', pa.float64(), nullable=False),
        pa.field('thrust', pa.float64(), nullable=False),
        pa.field('sphericality', pa.float64(), nullable=False),
        pa.field('part_px', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_py', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_pz', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_energy', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_mass', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_charge', pa.large_list(pa.field('item', pa.int32(), nullable=False)), nullable=False)
    ])

    data = SetAKArr(data_in_filepath)

    # Create a DataFrame, converting numpy arrays to lists for pandas compatibility
    df = pd.DataFrame({
        key: [arr.tolist() if isinstance(arr, (list, np.ndarray)) else arr for arr in val]  # Handle list or ndarray
        for key, val in data.items()
    })

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, parquet_out_filepath)
    logging.info(f"Written parquet file: {parquet_out_filepath}")


# Example usage
readFile('../raw_data/train.txt', '../data/Bmeson/train_file.parquet')
readFile('../raw_data/test.txt', '../data/Bmeson/test_file.parquet')
# readFile('../raw_data/testing.txt', '../data/Bmeson/testing_file.parquet')
