import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import logging
from sklearn.decomposition import PCA

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


import numpy as np
from sklearn.decomposition import PCA

def get_jetiness(px, py, pz, energy):
    """
    Compute jetiness for a collection of particles using the thrust axis.

    Parameters:
    - px, py, pz: Lists or arrays of particle momenta components.
    - energy: List or array of particle energies.

    Returns:
    - jetiness: Jetiness value, a measure of the collimation of the event.
    """
    
    # Convert inputs to numpy arrays
    px = np.array(px)
    py = np.array(py)
    pz = np.array(pz)
    energy = np.array(energy)

    # If there is only one particle, treat it as a pencil-like (strong jet-like) event
    if len(px) == 1:
        # In this case, the event is a single particle, which is essentially collimated
        # Jetiness is very small (ideally zero), as there is no spread
        # print("Single particle event - treated as strong jet/pencil-like.")
        return 0.0  # Jetiness is zero for a single particle (no spread)

    # Step 1: Calculate the thrust axis using PCA (Principal Component Analysis)
    # Combine the momentum components into a single matrix (rows: particles, columns: px, py, pz)
    momenta = np.vstack((px, py, pz)).T
    
    # Apply PCA to find the thrust axis (direction of maximum momentum concentration)
    pca = PCA(n_components=1)
    pca.fit(momenta)
    
    # The first principal component is the thrust axis (normalize it)
    thrust_axis = pca.components_[0]
    thrust_axis = thrust_axis / np.linalg.norm(thrust_axis)  # Normalize the thrust axis

    # Step 2: Calculate jetiness using the thrust axis
    # Compute transverse momentum for each particle
    pt = np.sqrt(px**2 + py**2)
    
    # Compute the pseudorapidity (eta) and azimuthal angle (phi) for each particle
    eta = 0.5 * np.log((energy + pz) / (energy - pz))
    phi = np.arctan2(py, px)
    
    # Step 3: Compute the distance (ΔR) between each particle and the thrust axis
    # The thrust axis is represented as a unit vector (thrust_axis)
    delta_phi = phi - np.arctan2(thrust_axis[1], thrust_axis[0])
    delta_eta = eta - np.arcsinh(thrust_axis[2])  # Use arcsinh for pseudorapidity
    delta_R = np.sqrt(delta_phi**2 + delta_eta**2)
    
    # Step 4: Compute jetiness: sum of pt * ΔR for all particles
    jetiness = np.sum(pt * delta_R)

    return jetiness


def get_thrust(px, py, pz):
    """
    Calculate thrust for a given set of particles based on their momenta (px, py, pz).
    
    Parameters:
    - px, py, pz: Arrays of particle momentum components in x, y, z directions.
    
    Returns:
    - thrust_value: The calculated thrust value.
    """
    # Convert to NumPy arrays
    px, py, pz = np.array(px), np.array(py), np.array(pz)

    # Remove zero-momentum particles to avoid division by zero or irrelevant calculations
    norms = np.sqrt(px**2 + py**2 + pz**2)
    mask = norms > 1e-15  # Remove particles with very small momentum
    px, py, pz, norms = px[mask], py[mask], pz[mask], norms[mask]

    if len(px) < 2:  # Handle the case where no valid particles are left
        thrust_value = 1
        return thrust_value

    # Create momenta array for PCA
    momenta = np.vstack((px, py, pz)).T

    # Apply PCA to find the thrust axis (direction of maximum momentum)
    pca = PCA(n_components=1)
    pca.fit(momenta)
    thrust_axis = pca.components_[0]

    # Normalize the thrust axis
    thrust_axis /= np.linalg.norm(thrust_axis)

    # Calculate thrust value: dot product of momenta with thrust axis, normalized by total momentum
    dot_products = np.abs(np.dot(momenta, thrust_axis))  # |p · n|
    total_momentum = np.sum(norms)  # ∑|p|
    thrust_value = np.sum(dot_products) / total_momentum

    return thrust_value

import numpy as np
from sklearn.decomposition import PCA

def get_sphericality(px, py, pz):
    """
    Calculate the sphericality of a system of particles based on their momenta.
    
    Parameters:
    - px, py, pz: Arrays of particle momentum components in x, y, z directions.
    
    Returns:
    - sphericality: The computed sphericality value (NaN for single-particle events).
    """
    # Convert to NumPy arrays
    px, py, pz = np.array(px), np.array(py), np.array(pz)

    # Remove zero-momentum particles to avoid division by zero or irrelevant calculations
    norms = np.sqrt(px**2 + py**2 + pz**2)
    mask = norms > 1e-15  # Remove particles with very small momentum
    px, py, pz, norms = px[mask], py[mask], pz[mask], norms[mask]

    if len(px) < 2:  # If there are fewer than 2 particles, sphericality is undefined
        return 0 # Sphericality doesn't apply for single-particle events

    # Compute the total momentum of the system
    total_momentum = np.sum(norms)

    # Build the momenta matrix for PCA
    momenta = np.vstack((px, py, pz)).T

    # Apply PCA to find the thrust axis (direction of maximum momentum)
    pca = PCA(n_components=1)
    pca.fit(momenta)
    thrust_axis = pca.components_[0]

    # Normalize the thrust axis
    thrust_axis /= np.linalg.norm(thrust_axis)

    # Project the momenta onto the thrust axis
    dot_products = np.abs(np.dot(momenta, thrust_axis))  # |p · n|
    
    # Calculate the sphericality value
    sphericality = 1 - 0.5 * (np.sum(dot_products) / total_momentum) ** 2
    
    return sphericality


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

                # Calculate thrust for the current event
                thrust_value = get_thrust(px_ls, py_ls, pz_ls)
                sph_value = get_sphericality(px_ls,py_ls,pz_ls)
                jetiness_value = get_jetiness(px_ls,py_ls,pz_ls,energy_ls)
                # if sph_value == 0:
                #     print(f"Thrust Value: {thrust_value}")
                #     print(f"Sphericality: {sph_value}")
                #     print(f"Jetiness: {jetiness_value}")
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
