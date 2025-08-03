import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def SetAKArr(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    n_particles_ls = []
    px = []
    py = []
    pz = []
    energy = []
    mass = []
    charge = []
    pdg = []

    px_ls = []
    py_ls = []
    pz_ls = []
    energy_ls = []
    mass_ls = []
    charge_ls = []
    pdg_ls = []

    _label1 = []
    _label2 = []
    _label3 = []
    _label4 = []
    _label5 = []

    # dirty = 0
    # first = 1
    n = 0
    #record the number of particles in one experiment
    for line in lines:
        if line.startswith('E'):
            if (not n == 0):
                n_particles_ls.append(n)
                #set up the larged list
                px.append(px_ls)
                py.append(py_ls)
                pz.append(pz_ls)
                energy.append(energy_ls)
                mass.append(mass_ls)
                charge.append(charge_ls)
                pdg.append(pdg_ls)
                
                px_ls = []
                px_ls = []
                py_ls = []
                pz_ls = []
                energy_ls = []
                charge_ls = []
                mass_ls = []
                pdg_ls = []

                exp_inf = line.split()
                _label1.append(float(exp_inf[1]))
                _label2.append(float(exp_inf[2]))
                _label3.append(float(exp_inf[3]))
                _label4.append(float(exp_inf[4]))

                # mass
                _label5.append(float(exp_inf[5]))
            else:
                exp_inf = line.split()
                _label1.append(float(exp_inf[1]))
                _label2.append(float(exp_inf[2]))
                _label3.append(float(exp_inf[3]))
                _label4.append(float(exp_inf[4])) 

                _label5.append(float(exp_inf[5]))
            n = 0
        else:
            #we ignore the photon
            par = line.split()
            ##particle +1
            n = n + 1
            px_ls.append(float(par[2]))
            py_ls.append(float(par[3]))
            pz_ls.append(float(par[4]))
            energy_ls.append(float(par[5]))
            mass_ls.append(float(par[6]))
            charge_ls.append(float(par[0]))

            pdg_ls.append(float(par[1]))

    px.append(px_ls)
    py.append(py_ls)
    pz.append(pz_ls)
    energy.append(energy_ls)
    mass.append(mass_ls)
    charge.append(charge_ls)
    pdg.append(pdg_ls)
    n_particles_ls.append(n)

    v = {}

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy
    v['part_mass'] = mass

    # for en_ls in energy:
        # total_energy = 0
        # for en in en_ls:
        #     total_energy += en
        # print(total_energy)

    v['part_charge'] = charge
    # v['label'] = np.stack((_label1, _label2, _label3, _label4, _label5), axis = -1)
    v['label'] = np.stack(_label5, axis = -1)
    
    for i in px:
        for j in i:
            if (np.isnan(j)):
                print('NaN here!')
    
    # v['label'] = _label5
    return v


def readFile(data_in_filepath, parquet_out_filepath):
    # Define the schema
    schema = pa.schema([
        pa.field('label', pa.float64(), nullable=False),
        pa.field('part_px', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_py', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_pz', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_energy', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
        pa.field('part_mass', pa.large_list(pa.field('item', pa.float64(), nullable=False)), nullable=False),
    ])

    data = SetAKArr(data_in_filepath)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Convert DataFrame to Table
    table = pa.Table.from_pandas(df, schema=schema)

    # Write to Parquet
    pq.write_table(table, parquet_out_filepath)

readFile('../raw_data/train.txt', '../data/Bmeson/train_file.parquet')
# readFile('../raw_data/val.txt', '../data/Bmeson/val_file.parquet')
readFile('../raw_data/test.txt', '../data/Bmeson/test_file.parquet')