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

    jet_pt = []
    nullarr = []
    nullarr_ls = []

    dirty = 0
    first = 1
    n = 0
    #record the number of particles in one experiment
    for line in lines:
        if line.startswith('E'):
            if (not n == 0 and not dirty):
                n_particles_ls.append(n)
                #set up the larged list
                px.append(px_ls)
                py.append(py_ls)
                pz.append(pz_ls)
                energy.append(energy_ls)
                mass.append(mass_ls)
                charge.append(charge_ls)
                pdg.append(pdg_ls)

                nullarr.append(nullarr_ls)
                nullarr_ls = []
                
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
                # _label5.append(1.0)
                jet_pt.append(float(exp_inf[1])+float(exp_inf[2])+float(exp_inf[3]))
            elif (n==0 and first):
                exp_inf = line.split()
                _label1.append(float(exp_inf[1]))
                _label2.append(float(exp_inf[2]))
                _label3.append(float(exp_inf[3]))
                _label4.append(float(exp_inf[4])) 

                _label5.append(float(exp_inf[5]))
                # _label5.append(1.0)
                jet_pt.append(float(exp_inf[1])+float(exp_inf[2])+float(exp_inf[3]))
            first = 0
            n = 0
            dirty = 0
        else:
            #we ignore the photon
            par = line.split()
            if (int(par[1]) == 22):
                dirty = 1
                pdg_ls = []
                px_ls = []
                py_ls = []
                pz_ls = []
                energy_ls = []
                charge_ls = []
                mass_ls = []

                nullarr_ls = []
            if (not dirty):
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
                nullarr_ls.append(1)

    px.append(px_ls)
    py.append(py_ls)
    pz.append(pz_ls)
    energy.append(energy_ls)
    mass.append(mass_ls)
    charge.append(charge_ls)
    pdg.append(pdg_ls)
    n_particles_ls.append(n)

    nullarr.append(nullarr_ls)

    v = {}
    #jet infor
    v['jet_pt'] = jet_pt
    v['jet_eta'] = _label1
    v['jet_phi'] = _label2
    v['jet_energy'] = _label4
    v['jet_mass'] = _label4
    v['jet_nparticles'] = n_particles_ls
    v['part_pid'] = pdg
    v['part_isNeutralHadron'] = nullarr
    v['part_isPhoton'] = nullarr
    v['part_isChargedHadron'] = nullarr
    v['part_isElectron'] = nullarr
    v['part_isMuon'] = nullarr


    v['part_deta'] = px
    v['part_dphi'] = py
    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy
    # v['part_mass'] = mass
    v['part_charge'] = charge
#     v['label'] = np.stack((_label1, _label2, _label3, _label4, _label5), axis = -1)
    v['label'] = np.stack(_label5, axis = -1)
    # print(v['label'].shape)
    # v['label'] = _label5
    return v


def readFile(data_in_filepath, parquet_out_filepath):
    # Define the schema
    schema = pa.schema([
        pa.field('label', pa.float64(), nullable=False),
        # pa.field('jet_pt', pa.float32(), nullable=False),
        # pa.field('jet_eta', pa.float32(), nullable=False),
        # pa.field('jet_phi', pa.float32(), nullable=False),
        # pa.field('jet_energy', pa.float32(), nullable=False),
        # pa.field('jet_mass', pa.float32(), nullable=False),
        # pa.field('jet_nparticles', pa.int64(), nullable=False),
        pa.field('part_px', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_py', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_pz', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_energy', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        # pa.field('part_deta', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        # pa.field('part_dphi', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_pid', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_isNeutralHadron', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_isPhoton', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_isChargedHadron', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_isElectron', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_isMuon', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
        pa.field('part_charge', pa.list_(pa.field('item', pa.float32(), nullable=False)), nullable=False),
    ])

    # # Example data
    # data = {
    #     'label': [1.0, 0.0],
    #     'jet_pt': [200.0, 300.0],
    #     'jet_eta': [0.5, -0.3],
    #     'jet_phi': [1.5, -2.1],
    #     'jet_energy': [500.0, 600.0],
    #     'jet_mass': [50.0, 60.0],
    #     'jet_nparticles': [3, 4],
    #     'part_px': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
    #     'part_py': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]],
    #     'part_pz': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]],
    #     'part_energy': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
    #     'part_deta': [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06, 0.07]],
    #     'part_dphi': [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06, 0.07]],
    #     'part_pid': [[11, 13, 22], [211, 321, 130, 310]],
    #     'part_isNeutralHadron': [[0, 0, 1], [0, 0, 1, 1]],
    #     'part_isPhoton': [[1, 1, 0], [0, 1, 0, 1]],
    #     'part_isChargedHadron': [[1, 0, 0], [1, 0, 1, 0]],
    #     'part_isElectron': [[1, 0, 0], [0, 1, 0, 0]],
    #     'part_isMuon': [[0, 1, 0], [0, 0, 1, 1]],
    #     'part_charge': [[-1, 1, 0], [1, -1, 0, 0]],
    # }

    data = SetAKArr(data_in_filepath)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Convert DataFrame to Table
    table = pa.Table.from_pandas(df, schema=schema)

    # Write to Parquet
    pq.write_table(table, parquet_out_filepath)

readFile('../raw_data/train.txt', '../data/Bmeson/train_file.parquet')
readFile('../raw_data/val.txt', '../data/Bmeson/val_file.parquet')
readFile('../raw_data/test.txt', '../data/Bmeson/test_file.parquet')