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

model_path='/expanse/lustre/projects/cwr109/icho1/SPICE/aimnet_models/aimnet2_b973c_ens.jpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.jit.load(model_path, map_location=device)

hartree2ev = 27.211386245988
ev2hartree = 1/hartree2ev
ev2kcalmol = 23.0605
np.set_printoptions(precision=10)
torch.set_printoptions(precision=10)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

profile = OrcaProfile(command='/home/icho1/orca_6_0_0_shared_openmpi416/orca')


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
    
    
def delete_files_if_exists(directory, file_prefix):
    # Construct the file paths
    original_file = os.path.join(directory, f"{file_prefix}.xyz")
    aminoacid_file = os.path.join(directory, f"{file_prefix}_aminoacid.xyz")
    ligand_file = os.path.join(directory, f"{file_prefix}_ligand.xyz")
    # print(aminoacid_file)
    # print(ligand_file)
    # Check if both the aminoacid and ligand files exist
    if os.path.isfile(original_file):
        os.remove(original_file)
        os.remove(aminoacid_file)
        os.remove(ligand_file)
        # print(f"Deleted files: {original_file}, {aminoacid_file}, {ligand_file}")
    else:
        # print(f"Files not found or conditions not met for: {file_prefix}")
        return

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





def ORCA_energy(mol_path, profile=profile:
    try:
        # Read the molecule
        molecule = read(mol_path)
        mol = next(pybel.readfile('xyz', mol_path))
        
        charge = mol.charge

        # Set up the ORCA calculator
        calculator = ORCA(
            label='orca',
            profile=profile,
            orcasimpleinput='b97-3c tightscf SCFConvForced slowconv',
            orcablocks='%PAL NPROCS 8 END',
            charge=charge,
            mult=1
        )

        # Assign the calculator to the molecule
        molecule.calc = calculator

        try:
            # Calculate the potential energy
            energy = molecule.get_potential_energy()

        except subprocess.CalledProcessError:
            print("ORCA calculation failed for mult=1")
            # Handle mult=2 if needed

        except Exception as e:
            print(f"An unexpected error occurred during ORCA calculation for {mol_path}: {e}")
            return None

        # Extract the final energy from the ORCA output file
        orca_out_path = os.path.join(os.getcwd(), 'orca.out')
        final_energy = None

        with open(orca_out_path, 'r') as file:
            for line in file:
                if "FINAL SINGLE POINT ENERGY" in line:
                    final_energy = float(line.split()[-1])
                    break

        if final_energy is None:
            raise ValueError("FINAL SINGLE POINT ENERGY not found in the ORCA output.")

        # Clean up generated files
        current_directory = os.getcwd()
        for file in glob.glob(os.path.join(current_directory, 'orca*')):
            os.remove(file)

        return final_energy

    except Exception as e:
        print(f"Error with ORCA calculation for {mol_path}: {e}")
        return None



def AIMNET2_energy(mol_path, model_path=model_path,ev2hartree=ev2hartree):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.jit.load(model_path, map_location=device)

    # Read the molecule file using Open Babel
    mol = next(pybel.readfile('xyz', mol_path))

    # Prepare input tensors for the model
    coord = torch.as_tensor([a.coords for a in mol.atoms]).unsqueeze(0).to(device)
    coord.requires_grad_(True)

    numbers = torch.as_tensor([a.atomicnum for a in mol.atoms]).unsqueeze(0).to(device)
    charge = torch.as_tensor([mol.charge]).to(device)

    _in = dict(
        coord=coord,
        numbers=numbers,
        charge=charge
    )

    _out = model(_in)

    energy_hartree = _out['energy'].item() * ev2hartree
    # print(energy_hartree)
    return energy_hartree

def process_group(file_path, group_name):
    with h5py.File(file_path, 'a') as f:
        group = f[group_name]

        lig_words, AA_words = group_name.split()
        
        if 'AIMNET2_b973c' in group:
            print(f"Skipping {group_name} as AIMNET2_b973c dataset already exists.")
            return
        
        # Read conformations and atomic numbers
        coord_complex = read_smiles_from_group(file_path, group_name, 'conformations')
        coord_AA = read_smiles_from_group(file_path, group_name, 'conformations_AA_ang')
        coord_lig = read_smiles_from_group(file_path, group_name, 'conformations_lig_ang')

        if len(coord_complex) == 0:
            return

        num_complex = read_smiles_from_group(file_path, group_name, 'atomic_numbers')
        num_AA = read_smiles_from_group(file_path, group_name, 'atomic_numbers_AA')
        num_lig = read_smiles_from_group(file_path, group_name, 'atomic_numbers_lig')

        # Create datasets
        datasets = {
            'AIMNET2_complex': group.create_dataset('AIMNET2_b973c', shape=(coord_complex.shape[0],), dtype='float64', chunks=True),
            'AIMNET2_AA': group.create_dataset('AIMNET2_b973c_AA', shape=(coord_complex.shape[0],), dtype='float64', chunks=True),
            'AIMNET2_lig': group.create_dataset('AIMNET2_b973c_lig', shape=(coord_complex.shape[0],), dtype='float64', chunks=True),
            'ORCA_complex': group.create_dataset('dft_total_energy_b973c', shape=(coord_complex.shape[0],), dtype='float64', chunks=True),
            'ORCA_AA': group.create_dataset('dft_total_energy_AA_b973c', shape=(coord_complex.shape[0],), dtype='float64', chunks=True),
            'ORCA_lig': group.create_dataset('dft_total_energy_lig_b973c', shape=(coord_complex.shape[0],), dtype='float64', chunks=True)
        }

        # Process each conformation
        for i in range(coord_complex.shape[0]):
            print(lig_words, AA_words, i)

            mol_path_comp = f"{lig_words}_{AA_words}{i}.xyz"
            mol_path_AA = f"{lig_words}_{AA_words}{i}_aminoacid.xyz"
            mol_path_lig = f"{lig_words}_{AA_words}{i}_ligand.xyz"

            save_xyz_file_complex(coord_complex[i], num_complex, mol_path_comp)
            save_xyz_file(coord_AA[i], num_AA, mol_path_AA)
            save_xyz_file(coord_lig[i], num_lig, mol_path_lig)

            # Compute energies and store them in datasets
            datasets['AIMNET2_complex'][i] = AIMNET2_energy(mol_path_comp)
            datasets['AIMNET2_AA'][i] = AIMNET2_energy(mol_path_AA)
            datasets['AIMNET2_lig'][i] = AIMNET2_energy(mol_path_lig)
            datasets['ORCA_complex'][i] = ORCA_energy(mol_path_comp)
            datasets['ORCA_AA'][i] = ORCA_energy(mol_path_AA)
            datasets['ORCA_lig'][i] = ORCA_energy(mol_path_lig)

            # Clean up
            delete_files_if_exists(os.getcwd(), f"{lig_words}_{AA_words}{i}")

def process_file(file_path):
    with h5py.File(file_path, 'a') as f:
        group_namelist = list(f.keys())

    for group_name in group_namelist:
        process_group(file_path, group_name)

file_path = "AminoAcidLigand_v2.1.hdf5"
process_file(file_path)
