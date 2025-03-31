import os 
import h5py
import glob
import numpy as np
from openbabel import openbabel as ob 
import torch
import ase
from ase.io import read, write
from ase.calculators.orca import ORCA
from ase.calculators.orca import OrcaProfile
from ase import Atoms
from openbabel import pybel
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import shutil
from rdkit.Chem import rdmolops
from tqdm import tqdm
import subprocess
from mace.calculators import mace_off
from numpy import printoptions

#model_path='/expanse/lustre/projects/cwr109/icho1/SPICE/aimnet_models/aimnet2_b973c_ens.jpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = torch.jit.load(model_path, map_location=device)

hartree2ev = 27.211386245988
ev2hartree = 1/hartree2ev
ev2kcalmol = 23.0605
np.set_printoptions(precision=10)
torch.set_printoptions(precision=10)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False




def save_xyz_file_complex(coord_ALAs, number_ALAs, save_filename):

    # Dictionary to map atomic numbers to element symbols
    atomic_number_to_symbol = {1: 'H',5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
                                     14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 
                                     33: 'As', 34: 'Se', 35: 'Br', 53: 'I'}

    # Conversion factor from Bohr to Angstrom
    bohr_to_angstrom = 0.529177249

    # Convert coordinates from Bohr to Angstrom
    coord_ALAs_ang = coord_ALAs * bohr_to_angstrom

    # Number of atoms
    num_atoms = len(coord_ALAs_ang)
    
    base_name = os.path.splitext(os.path.basename(save_filename))[0]

    # Creating the XYZ format content
    xyz_content = f"{num_atoms}\n"
    xyz_content += f"{base_name}\n"


    # Adding atomic symbols and coordinates to the XYZ content
    for num, coord in zip(number_ALAs, coord_ALAs_ang):
        element_symbol = atomic_number_to_symbol[num]
        xyz_content += f"{element_symbol} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}\n"

    # Writing to an XYZ file
    with open(save_filename, "w") as xyz_file:
        xyz_file.write(xyz_content)
        

def explore_hdf5_file(file_path, max_examples=5):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Print the root directory
        print(f"Root directory: {file_path}")
        print("")

        # Function to explore groups and print examples
        def explore_groups(group, indent=0, examples_limit=5):
            group_count = 0
            examples_printed = 0

            # Collect group keys
            group_keys = list(group.keys())
            print(f"{'  ' * indent}Total groups: {len(group_keys)}")
            print(f"{'  ' * indent}Group names: {', '.join(group_keys)}")
            print("")

            # Explore and print example groups
            for key in group_keys:
                if examples_printed >= examples_limit:
                    break
                
                item = group[key]
                if isinstance(item, h5py.Group):
                    print(f"{'  ' * indent}Group: {key}/")
                    explore_groups(item, indent + 1, examples_limit - examples_printed)
                    examples_printed += 1
                    group_count += 1
                elif isinstance(item, h5py.Dataset):
                    shape = item.shape
                    size = item.size
                    print(f"{'  ' * indent}Dataset: {key} (size: {size}, shape: {shape})")
                    
                    # Calculate and print the number of conformations if shape is known
                    if shape and len(shape) == 3 and shape[1] == 3:
                        num_conformations = size // (shape[0] * shape[1])
                        print(f"{'  ' * (indent + 1)}Number of conformations: {num_conformations}")

            return group_count

        # Start exploring from root
        total_groups = explore_groups(f, examples_limit=max_examples)
        print(f"Total number of groups: {total_groups}")
        
def read_smiles_from_group(file_path, group_name, dataset_name):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Navigate to the specified group
        group = f[group_name]
        
        # Read the specified dataset within the group
        dataset = group[dataset_name]
        # dataset2= group[dataset_name2]
        
        # Get the data from the dataset
        smiles_data = dataset[()]
        # smiles_data2=dataset2[()]
        
        return smiles_data
    
def compute_mace(mol_path,ev2hartree=ev2hartree):
    molecule = read(mol_path)
    calc = mace_off(model="large", device='cuda',default_dtype="float64")

    molecule.calc = calc
    return(molecule.get_potential_energy()*ev2hartree)

def read_xyz_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            atom_type = parts[0]
            # Round the coordinates to 5 decimal places
            # x, y, z = [round(float(coord), 3) for coord in parts[1:]]
            x, y, z = [float(f"{float(coord):.2f}") for coord in parts[1:]]
            atoms.append((atom_type, (x, y, z)))
    return atoms


def save_xyz_file(coord_ALAs, number_ALAs, save_filename):

    # Dictionary to map atomic numbers to element symbols
    atomic_number_to_symbol = {1: 'H',5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
                                     14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 
                                     33: 'As', 34: 'Se', 35: 'Br', 53: 'I'}

    # Number of atoms
    num_atoms = len(coord_ALAs)
    
    base_name = os.path.splitext(os.path.basename(save_filename))[0]

    # Creating the XYZ format content
    xyz_content = f"{num_atoms}\n"
    xyz_content += f"{base_name}\n"


    # Adding atomic symbols and coordinates to the XYZ content
    for num, coord in zip(number_ALAs, coord_ALAs):
        element_symbol = atomic_number_to_symbol[num]
        xyz_content += f"{element_symbol} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}\n"

    # Writing to an XYZ file
    with open(save_filename, "w") as xyz_file:
        xyz_file.write(xyz_content)
        

def find_indices(original_atoms, subset_atoms, tolerance=0.5):
    def is_close(atom1, atom2, tol):
        # Check if atom types are the same
        if atom1[0] != atom2[0]:
            return False
        # Check if the coordinates are within the tolerance
        for coord1, coord2 in zip(atom1[1], atom2[1]):
            # print(coord1)
            # print(coord2)
            if abs(coord1 - coord2) > tol:
                return False
        return True
    
    indices = []
    for subset_atom in subset_atoms:
        for i, original_atom in enumerate(original_atoms):
            if is_close(original_atom, subset_atom, tolerance):
                indices.append(i + 1)  # Add 1 to make it 1-indexed
                break
    return indices



def process_group2(file_path, group_name):
    with h5py.File(file_path, 'a') as f:
        group = f[group_name]

        lig_words, AA_words = group_name.split()

        # Read conformations and atomic numbers
        coord_complex = read_smiles_from_group(file_path, group_name, 'conformations')
        
        if len(coord_complex) == 0:
            return
        
        dft_mace = group.create_dataset("MACE_wb97", shape=(coord_complex.shape[0],), maxshape=(coord_complex.shape[0],), dtype='float64', chunks=True)
        #dft_mace_AA = group.create_dataset("MACE_wb97_AA", shape=(coord_complex.shape[0],), maxshape=(coord_complex.shape[0],), dtype='float64', chunks=True)
        #dft_mace_lig = group.create_dataset("MACE_wb97_lig", shape=(coord_complex.shape[0],), maxshape=(coord_complex.shape[0],), dtype='float64', chunks=True)
        
        
        coord_AA = read_smiles_from_group(file_path, group_name, 'conformations_AA_ang')
        coord_lig = read_smiles_from_group(file_path, group_name, 'conformations_lig_ang')
        num_complex = read_smiles_from_group(file_path, group_name, 'atomic_numbers')
        num_AA = read_smiles_from_group(file_path, group_name, 'atomic_numbers_AA')
        num_lig = read_smiles_from_group(file_path, group_name, 'atomic_numbers_lig')

        coord_mbis = read_smiles_from_group(file_path, group_name, 'mbis_charges')
        
        
        # b973c_total = read_smiles_from_group(file_path, group_name, 'dft_total_energy_b973c')
        # b973c_AA = read_smiles_from_group(file_path, group_name, 'dft_total_energy_AA_b973c')
        # b973c_lig = read_smiles_from_group(file_path, group_name, 'dft_total_energy_lig_b973c')
        
        # b973c_total_aimnet2 = read_smiles_from_group(file_path, group_name, 'AIMNET2_b973c')
        # b973c_AA_aimnet2 = read_smiles_from_group(file_path, group_name, 'AIMNET2_b973c_AA')
        # b973c_lig_aimnet2 = read_smiles_from_group(file_path, group_name, 'AIMNET2_b973c_lig')
        
        for i in range(coord_complex.shape[0]):
                
            mol_path_comp = f"{lig_words}_{AA_words}{i}.xyz"
            mol_path_AA = f"{lig_words}_{AA_words}{i}_aminoacid.xyz"
            mol_path_lig = f"{lig_words}_{AA_words}{i}_ligand.xyz"
  
            save_xyz_file_complex(coord_complex[i], num_complex, mol_path_comp)
        
            if lig_words== "0CM":
                save_xyz_file(coord_AA[i], num_AA[i], mol_path_AA)
                save_xyz_file(coord_lig[i], num_lig[i], mol_path_lig)
            elif lig_words == "HH2":
                save_xyz_file(coord_AA[i], num_AA[i], mol_path_AA)
                save_xyz_file(coord_lig[i], num_lig[i], mol_path_lig)
            else:
                save_xyz_file(coord_AA[i], num_AA, mol_path_AA)
                save_xyz_file(coord_lig[i], num_lig, mol_path_lig)
            
                 
            comp_Mbis = int(round(np.sum(coord_mbis[i]), 0))

     
            original_atoms = read_xyz_file(f"{lig_words}_{AA_words}{i}.xyz")
            amino_atoms = read_xyz_file(f"{lig_words}_{AA_words}{i}_aminoacid.xyz")
            ligand_atoms = read_xyz_file(f"{lig_words}_{AA_words}{i}_ligand.xyz")
    
            amino_indices = find_indices(original_atoms, amino_atoms)
            ligand_indices = find_indices(original_atoms, ligand_atoms)
    
            amino_indices_0_based = np.array([i - 1 for i in amino_indices])
            ligand_indices_0_based = np.array([i - 1 for i in ligand_indices])
                
            comp_Mbis = int(round(np.sum(coord_mbis[i]), 0))
                
            amino_Mbis = coord_mbis[i][amino_indices_0_based]
            amino_Mbis = int(round(np.sum(amino_Mbis), 0))
        
            ligand_Mbis = coord_mbis[i][ligand_indices_0_based]
            ligand_Mbis = int(round(np.sum(ligand_Mbis), 0))
        
                                
            if comp_Mbis == 0:
                if len(amino_indices) + len(ligand_indices) == len(num_complex):
                    # print(lig_words, AA_words,i)
                    if amino_Mbis==0 and ligand_Mbis==0:

                        print(lig_words, AA_words,i)
                        dft_mace[i]=compute_mace(mol_path_comp)
                        #dft_mace_AA[i]=compute_mace(mol_path_AA)
                        #dft_mace_lig[i]=compute_mace(mol_path_lig)
                        
                        group['MACE_wb97'][i] = dft_mace[i]
                        #group['MACE_wb97_AA'][i] = dft_mace_AA[i]
                        #group['MACE_wb97_lig'][i] = dft_mace_lig[i]
                                
            else: 
                print ("molecule is not neutral",comp_Mbis,lig_words,AA_words)
            
            os.remove(mol_path_comp)
            os.remove(mol_path_AA)
            os.remove(mol_path_lig)
            
                    
                
def process_file2(file_path):
    with h5py.File(file_path, 'a') as f:
        group_namelist = list(f.keys())

    for group_name in group_namelist:
        process_group2(file_path, group_name)


file_path = "AminoAcidLigand_v2.1.hdf5"


process_file2(file_path)

