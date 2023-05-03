import numpy as np
from .S_parser import get_graph_data,file_reader
import torch
def get_data(pdb_file, rscb=False, uniprot=False):
    '''pdb_file: absolute or relative file_path to pdb;
                pdb name if rscb is True;
                uniprot id if uniprot is True;
        returns dictionary of parsed atomic coordinate where keys are chain IDS and Chains
    '''
    vertex_dict = get_graph_data(pdb_file, rscb=rscb, uniprot=uniprot)
    return vertex_dict, list(vertex_dict.keys())

def get_vertex_edges(vertex_dict, chain):
    try:
        vertices = []
        if isinstance(chain, str):
            vertices = vertex_dict[chain]
        else:
            for ch in chain:
                vertices += vertex_dict[ch]
        information=np.asanyarray(vertices)[:,[0,1,2,6,7,8,3,4,5]]
        vertex_coords = np.asarray(vertices)[:, 9:]
        positions = np.asarray(vertices)[:, 3:6]
        positions[positions == 'None'] = 0
        positions = np.asarray(positions, float)
        positions = np.nan_to_num(positions)
        vertex_coords = np.asarray(vertex_coords, float)
        mask = np.where(positions.any(axis=1))[0]
        return vertex_coords[mask], positions[mask],information[mask]
    except KeyError as e:
        print(f'{e} chain does not exist')
        return
def get_HET_ATOM(pdb_file,ligand_name,rscb=False):
    '''pdb_file : pdb_file path
                : if rscb True then pdb name
       ligand_name: list of ligand_name or single ligand_name (uppercase)

       returns : Dictionary of ligands coordinates where keys are HETATOM_no:HETATOM_chain:HETATOM_Name
    '''
    if isinstance(ligand_name,str):
        ligand_name=[ligand_name]
    ligand_name=[x.strip() for x in ligand_name]
    het_coords={}
    for line in file_reader(pdb_file,rscb=rscb):
        if line[:6].strip()=='HETATM':
            if line[17:20].strip() in ligand_name:
                chain=line[21].strip()
                res_no=line[22:26].strip()
                atom_x=line[30:38].strip()
                atom_y=line[38:46].strip()
                atom_z=line[46:54].strip()
                if f'{res_no}:{chain}:{line[17:20].strip()}' in het_coords.keys():
                    het_coords[f'{res_no}:{chain}:{line[17:20].strip()}'].append([float(atom_x),float(atom_y),float(atom_z)])
                else:
                    het_coords[f'{res_no}:{chain}:{line[17:20].strip()}']=[[float(atom_x),float(atom_y),float(atom_z)]]
    return het_coords
def create_sparse_tensor(coordinates, features,information):
    coordinates -= np.min(coordinates, axis=0)
    coordinates = coordinates.round().astype(int)
    _, indices = np.unique(coordinates, axis=0, return_index=True)
    features = features[indices, :]
    coordinates = coordinates[indices, :]
    information=information[indices,::]
    return coordinates, features,information

def get_coordinates_features(vertex_dict, chain=None):
    '''
    chain:      List of chains or single chain as string (case sensitive)
                default : None (Considers all chains)

    Returns coordinates and features and information related to each atoms
    '''
    # vertex_dict, chains = get_data(pdb_file, rscb, uniprot)
    vertices, positions,information = get_vertex_edges(vertex_dict, chain)
    coordinates, features,information = create_sparse_tensor(positions, vertices,information)
    return torch.from_numpy(coordinates), torch.from_numpy(features),information
