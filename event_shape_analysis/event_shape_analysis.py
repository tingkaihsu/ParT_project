import numpy as np
# import math
# from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt

def event_shape_analysis(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        # Vertex ID record the what vertex is currently
        parent_vertex_id = 0
        B_plus_final_states = []
        B_minus_final_states = []
        
        B_plus_vertices = []
        B_minus_vertices = []
        
        for line in lines:
            # New event
            if line.startswith('E'):
                parent_vertex_id = 0
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
                        abs(int(partcl_dt[11])),               # Child vertex
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
        
        print(B_plus_final_states)
        # print(B_minus_vertices)
        return
    
event_shape_analysis("hepMCtest")