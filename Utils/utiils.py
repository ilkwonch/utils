import os 
import h5py
import glob
import numpy as np
from openbabel import openbabel as ob 
import numpy as np 
import torch
from ase.io import read
from ase.calculators.orca import ORCA
from openbabel import pybel
import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import shutil
import argparse


np.set_printoptions(precision=8)
torch.set_printoptions(precision=8)

def is_amino_acid(molecule, smarts_pattern, AA_name):
    """Check if the molecule matches the SMARTS pattern for amino acids and additional conditions."""
    # Check SMARTS pattern
    pattern = ob.OBSmartsPattern()
    pattern.Init(smarts_pattern)
    if not pattern.Match(molecule):
        return False

    # Check molecular weight range
    min_weight = 120
    max_weight = 250
    if not (min_weight <= molecule.GetMolWt() <= max_weight):
        return False

    # Check elemental composition
    if AA_name in ("CYS", "MET"):
        elements_in_molecule = {6, 1, 7, 8, 16}  # C, H, N, O, S
        atomic_nums_in_molecule = set(atom.GetAtomicNum() for atom in ob.OBMolAtomIter(molecule))
        if not atomic_nums_in_molecule.issubset(elements_in_molecule):
            return False
    else:
        elements_in_molecule = {6, 1, 7, 8}  # C, H, N, O,
        atomic_nums_in_molecule = set(atom.GetAtomicNum() for atom in ob.OBMolAtomIter(molecule))
        if not atomic_nums_in_molecule.issubset(elements_in_molecule):
            return False
        
    return True

def initialize_smarts_patterns_PRO():
    """Initialize SMARTS patterns for amino acid recognition."""
    # Define SMARTS patterns for different structures
    smarts_patterns = {
        "alanine_1": "C1CC(NC1)C(=O)NC",       # First alanine pattern
        "alanine_2": "C1CN(C(=O)NC)CC1",        # Second alanine pattern
        "alanine_3": "C1CCC(C(=O)NC)N1",        # Third alanine pattern
        "alanine_4" : "C1CN(C(=O)C)CC1"
    }
    
    return smarts_patterns   

def initialize_smarts_patterns():
    """Initialize SMARTS patterns for amino acid recognition."""
    # Define SMARTS patterns for different structures
    
    # smarts_patterns = "C(C(=O)NC)NC(C)=O"
    smarts_patterns =  "C(C(=O)NC)(NC(C)=O)C"
    return smarts_patterns
    
def is_lig_smiles(molecule, smarts_pattern):
    if smarts_pattern == None:
        return False
    else:
        pattern = ob.OBSmartsPattern()
        pattern.Init(smarts_pattern)
        return pattern.Match(molecule) > 0


    
    return smarts_patterns
        
def remove_xyz_files(directory):


    # Construct the search pattern for .xyz files
    search_pattern = os.path.join(directory, '*.xyz')
    
    # List all .xyz files in the directory
    xyz_files = glob.glob(search_pattern)
    
    # Remove each .xyz file
    for file_path in xyz_files:
        try:
            os.remove(file_path)
            # print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
            
def initialize_obconversion():
    """Initialize Open Babel conversion object for .xyz files."""
    obconv = ob.OBConversion()
    obconv.SetInAndOutFormats("xyz", "xyz")
    return obconv


def process_xyz_files(input_folder, aminoacid_folder, ligand_folder, smarts_pattern, lig_name, AA_name):
    obconv = initialize_obconversion()

    # Ensure output directories exist
    if not os.path.exists(aminoacid_folder):
        # print(f"Creating directory: {aminoacid_folder}")
        os.makedirs(aminoacid_folder, exist_ok=True)
    if not os.path.exists(ligand_folder):
        # print(f"Creating directory: {ligand_folder}")
        os.makedirs(ligand_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".xyz"):
            file_path = os.path.join(input_folder, filename)
            cluster = ob.OBMol()
            success = obconv.ReadFile(cluster, file_path)
            if not success:
                # print(f"Failed to read XYZ file: {file_path}")
                continue

            mols = list(cluster.Separate())
            # print(f"Number of molecules separated: {len(mols)}")
            if len(mols)==2:
                if sum(is_amino_acid(mol, smarts_pattern,AA_name) for mol in mols) ==1:  #uniqAA-uniq lig (checked)
                    
                    for idx, mol in enumerate(mols):
                        if is_amino_acid(mol, smarts_pattern,AA_name):
                            aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                            try:
                                obconv.WriteFile(mol, aminoacid_path)
                                # print(f"Written amino acid {idx + 1} to {aminoacid_path}")
                            except Exception as e:
                                print(f"Failed to write amino acid file: {aminoacid_path}. Error: {e}")
                        else:
                            ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                            try:
                                obconv.WriteFile(mol, ligand_path)
                                # print(f"Written ligand {idx + 1} to {ligand_path}")
                            except Exception as e:
                                print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                else:
                    if (AA_name == "TRP" and sum(mol.GetFormula()=="C14H17N3O2" for mol in mols)):
                        for idx, mol in enumerate(mols):
                            if mol.GetFormula()=="C14H17N3O2":
                                aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                try:
                                    obconv.WriteFile(mol, aminoacid_path)
                                except Exception as e:
                                    print(f"Failed to write amino acid file: {aminoacid_path}. Error: {e}")
                            else:
                                ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                try:
                                    obconv.WriteFile(mol, ligand_path)
                                except Exception as e:
                                    print(f"Failed to write ligand file: {ligand_path}. Error: {e}")

          
                    if sum(is_ligand(mol,AA_name) for mol in mols) == 1: #check unique types of atom elements,(checked)
                        for idx, mol in enumerate(mols):
                            if is_ligand(mol,AA_name):
                                ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                try:
                                    obconv.WriteFile(mol, ligand_path)
                                except Exception as e:
                                    print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                            else:
                                aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                try:
                                    obconv.WriteFile(mol, aminoacid_path)
                                except Exception as e:
                                    print(f"Failed to write amino acid file: {aminoacid_path}. Error: {e}")
                    else:
                        if sum(count_NO_in_molecule(mol,AA_name) for mol in mols) == 1:
                            for i, mol in enumerate(mols):
                                if count_NO_in_molecule(mol,AA_name):
                                    aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                    try:
                                        obconv.WriteFile(mol, aminoacid_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                                else:
                                    ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                    try:
                                        obconv.WriteFile(mol, ligand_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                                        
                        lig_smarts_pattern="C1NO1"
                        if sum(is_amino_acid(mol, lig_smarts_pattern,AA_name) for mol in mols) == 1:
                            for i, mol in enumerate(mols):
                                if is_amino_acid(mol, lig_smarts_pattern,AA_name):
                                    aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                    try:
                                        obconv.WriteFile(mol, aminoacid_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                                else:
                                    ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                    try:
                                        obconv.WriteFile(mol, ligand_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {ligand_path}. Error: {e}")          
                    
                        else:                                                                                                                                                                               
                            lig_smarts_pattern = get_smiles_from_ligand_name(lig_name)      #check exact smiles_pattern of ligand

                            for i, mol in enumerate(mols):                                                                                                            
                                if is_lig_smiles(mol, lig_smarts_pattern):
                                    ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                    try:
                                        obconv.WriteFile(mol, ligand_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                                else:
                                    aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                    try:
                                        obconv.WriteFile(mol, aminoacid_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")

            else:
                if (AA_name == "TRP" and sum(mol.GetFormula()=="C14H17N3O2" for mol in mols)):
                    non_amino_acids = ob.OBMol()
                    for idx, mol in enumerate(mols):
                        if mol.GetFormula()=="C14H17N3O2":
                            aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                            try:
                                obconv.WriteFile(mol, aminoacid_path)
                            except Exception as e:
                                print(f"Failed to write amino acid file: {aminoacid_path}. Error: {e}")
                        else:
                            non_amino_acids += mol
                        ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                        try:
                            obconv.WriteFile(non_amino_acids, ligand_path)
                        except Exception as e:
                            print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                
                else:
                    non_amino_acids = ob.OBMol()
                    if sum(is_amino_acid(mol, smarts_pattern,AA_name) for mol in mols) ==1:                   
                        for idx, mol in enumerate(mols):
                            if is_amino_acid(mol, smarts_pattern,AA_name): # uniqAA- uniq Lig( more than 2 clusters, checked)
                                aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                try:
                                    obconv.WriteFile(mol, aminoacid_path)
                                    # print(f"Written amino acid {idx + 1} to {aminoacid_path}")
                                except Exception as e:
                                    print(f"Failed to write amino acid file: {aminoacid_path}. Error: {e}")
                            else:
                                # ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                # try:
                                #     obconv.WriteFile(mol, ligand_path)
                                #     # print(f"Written ligand {idx + 1} to {ligand_path}")
                                # except Exception as e:
                                #     print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                                    
                                non_amino_acids += mol
                            ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                            try:
                                obconv.WriteFile(non_amino_acids, ligand_path)
                            except Exception as e:
                                print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                    else:
                        
                        lig_smarts_pattern="C1NO1"
                        non_amino_acids = ob.OBMol()
                        if sum(is_amino_acid(mol, lig_smarts_pattern,AA_name) for mol in mols) == 1:
                            for i, mol in enumerate(mols):
                                if is_amino_acid(mol, lig_smarts_pattern,AA_name):
                                    aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                    try:
                                        obconv.WriteFile(mol, aminoacid_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                                else:
                                    non_amino_acids += mol
                                ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                        
                                try:
                                    obconv.WriteFile(non_amino_acids, ligand_path)
                                except Exception as e:
                                    print(f"Failed to write ligand file: {ligand_path}. Error: {e}")                       
                                        
                        if sum(is_ligand(mol,AA_name) for mol in mols) == 1: # Accumulate non-amino acids/check unique types of atom elements ( more than 2 clusters, )
                            non_amino_acids = ob.OBMol()
                            for idx, mol in enumerate(mols):
                                if is_ligand(mol,AA_name):
                                    ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                    try:
                                        obconv.WriteFile(mol, ligand_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                                else:
                                    non_amino_acids += mol
                            aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                            try:
                                obconv.WriteFile(non_amino_acids, aminoacid_path)
                            except Exception as e:
                                print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                        else:
                            # if sum(count_NO_in_molecule(mol,AA_name) for mol in mols) == 1:

                            # Accumulate non-amino acids via specifying exact ligand smiles code( more than 2 clusters, checked )
                            lig_smarts_pattern = get_smiles_from_ligand_name(lig_name)
                            non_amino_acids = ob.OBMol()
                            molecular_formula = None
                        
                            if lig_smarts_pattern:
                                mol_pattern = Chem.MolFromSmiles(lig_smarts_pattern)
                                molecular_formula = rdMolDescriptors.CalcMolFormula(mol_pattern)
                        
                            for idx, mol in enumerate(mols):
                                # if (is_lig_smiles(mol, lig_smarts_pattern) ) or (molecular_formula == mol.GetFormula()):
                                if (lig_smarts_pattern and is_lig_smiles(mol, lig_smarts_pattern)) or (molecular_formula and molecular_formula == mol.GetFormula()):
    
                                    ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                    try:
                                        obconv.WriteFile(mol, ligand_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                                else:
                                    non_amino_acids += mol
                            
                            aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                            try:
                                obconv.WriteFile(non_amino_acids, aminoacid_path)
                            except Exception as e:
                                print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                            
                            
                        
def is_ligand(molecule,AA_name):   
    if AA_name in ("CYS", "MET"):
        allowed_elements = {5, 9, 14, 15, 17, 33, 34, 35, 53}
        atomic_nums_in_molecule = set(atom.GetAtomicNum() for atom in ob.OBMolAtomIter(molecule))
        if atomic_nums_in_molecule.isdisjoint(allowed_elements): #any overlap?
            return False
    else:
        allowed_elements = {5, 9, 14, 15, 16, 17, 33, 34, 35, 53} #S included
        atomic_nums_in_molecule = set(atom.GetAtomicNum() for atom in ob.OBMolAtomIter(molecule))
        if atomic_nums_in_molecule.isdisjoint(allowed_elements): #any overlap?
            return False
    return True
    
    

def get_smiles_from_ligand_name(ligand_name):
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_name}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        smiles = data.get('rcsb_chem_comp_descriptor', {}).get('smiles', None)
        if smiles:
            return smiles
        else:
            raise ValueError(f"No SMILES found for {ligand_name}")
    else:
        return None
    
def count_NO_in_molecule(mol, AA_name="XXX"):
    # Predefined initial nitrogen and oxygen counts for each amino acid (using three-letter codes)
    AA_count = {
    "GLY": (2, 2),
    "ALA": (2, 2),
    "VAL": (2, 2),
    "LEU": (2, 2),
    "ILE": (2, 2),
    "MET": (2, 2),
    "PRO": (3, 2),
    "PHE": (2, 2),
    "TYR": (2, 3),
    "TRP": (3, 2),
    "SER": (2, 3),
    "THR": (2, 3),
    "CYS": (2, 2),
    "ASN": (3, 3),
    "GLN": (3, 3),
    "ASP": (2, 4),
    "GLU": (2, 4),
    "LYS": (3, 2),
    "ARG": (5, 2),
    "HIS": (4, 2),
    "XXX": (0, 0)
    }

    # Get the initial nitrogen and oxygen counts for the amino acid
    # initial_nitrogens, initial_oxygens = initial_counts.get(AA_name, (0, 0))

    num_nitrogens = 0
    num_oxygens = 0

    # Count the number of nitrogen and oxygen atoms in the molecule
    for atom in ob.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 7:  # Nitrogen
            num_nitrogens += 1
        elif atomic_num == 8:  # Oxygen
            num_oxygens += 1
    if not (num_nitrogens,num_oxygens) == (AA_count.get(AA_name,(0,0))):
        return False
    return True

    
    

def delete_files_if_exists(directory, file_prefix):
    # Construct the file paths
    original_file = os.path.join(directory, f"{file_prefix}.xyz")
    aminoacid_file = os.path.join(directory, "amino", f"{file_prefix}_aminoacid.xyz")
    ligand_file = os.path.join(directory, "ligand", f"{file_prefix}_ligand.xyz")
    # print(aminoacid_file)
    # print(ligand_file)
    # Check if both the aminoacid and ligand files exist
    if os.path.isfile(aminoacid_file) and os.path.isfile(ligand_file):
        
        if os.path.isfile(original_file):
            os.remove(original_file)
        os.remove(aminoacid_file)
        os.remove(ligand_file)
        # print(f"Deleted files: {original_file}, {aminoacid_file}, {ligand_file}")
    else:
        # print(f"Files not found or conditions not met for: {file_prefix}")
        return


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

def read_xyz_file2(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0].strip())  # First line contains the number of atoms
    comment = lines[1].strip()         # Second line is a comment line
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            atom_type = parts[0]
            # x, y, z = [round(float(coord), 3) for coord in parts[1:]]
            x, y, z = [float(f"{float(coord):.10f}") for coord in parts[1:]]
            atoms.append((atom_type, (x, y, z)))
    return num_atoms, comment, atoms



def write_xyz_file(file_path, comment, atoms):
    with open(file_path, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom_type, (x, y, z) in atoms:
            f.write(f"{atom_type} {x:.5f} {y:.5f} {z:.5f}\n")


def separate_xyz(input_file, amino_indices, ligand_indices, output_prefix):
    # Read the input file
    num_atoms, comment, atoms = read_xyz_file2(input_file)
    
    # Separate atoms into amino acid and ligand based on indices
    amino_atoms = [atoms[i-1] for i in amino_indices]  # Indices are 1-based, so subtract 1
    ligand_atoms = [atoms[i-1] for i in ligand_indices] 
    
    # Write the separated atoms to new files
    write_xyz_file(f"amino/{output_prefix}_aminoacid.xyz", comment, amino_atoms)
    write_xyz_file(f"ligand/{output_prefix}_ligand.xyz", comment, ligand_atoms)
    
    
def assign_coord_number(file_path):

    # Dictionary to map atomic numbers to element symbols
    atomic_number_to_symbol = {1: 'H',5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
                                     14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 
                                     33: 'As', 34: 'Se', 35: 'Br', 53: 'I'}
    

    # Reverse dictionary to map element symbols to atomic numbers
    symbol_to_atomic_number = {v: k for k, v in atomic_number_to_symbol.items()}

    # Read the XYZ file
    with open(file_path, "r") as xyz_file:
        lines = xyz_file.readlines()

    # Number of atoms from the first line
    num_atoms = int(lines[0].strip())

    # Initialize lists to store atomic numbers and coordinates
    number_AA = []
    coord_AA = []

    # Process each line after the first two lines
    for line in lines[2:num_atoms + 2]:
        parts = line.split()
        element_symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

        # Convert element symbol to atomic number
        atomic_number = symbol_to_atomic_number[element_symbol]

        # Append to lists
        number_AA.append(atomic_number)
        coord_AA.append([x, y, z])

    # Convert lists to numpy arrays with specified data types
    number_AA_array = np.array(number_AA, dtype=np.int16)
    coord_AA_array = np.array(coord_AA, dtype=np.float64)

    return number_AA_array, coord_AA_array


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
        
        
def has_valid_data(coords):
    coords = np.array(coords)  # Ensure coords is a NumPy array
    return not np.all(np.all(coords == [0., 0., 0.], axis=1))

def find_valid_index(coord_list, start_index):
    for i in range(start_index, len(coord_list)):
        if has_valid_data(coord_list[i]):
            return i    
    return None
    
def check_len_mols(file_path):
    obconv = initialize_obconversion()
    cluster = ob.OBMol()
    success = obconv.ReadFile(cluster, file_path)
    mols = list(cluster.Separate())
            
    return len(mols)

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

def update_dataset(f, dataset_name, shape, maxshape,dtype='float64'):
    if dataset_name in f:
        del f[dataset_name]  # Delete the existing dataset if it exists
    f.create_dataset(dataset_name, shape=shape, maxshape=maxshape,dtype=dtype, chunks=True)
    
    
    
def is_amino_acid2(molecule, smarts_patterns, AA_name):
    """Check if the molecule matches the criteria for being an amino acid."""
    
    # Check if the molecule matches any of the SMARTS patterns
    for name, smarts in smarts_patterns.items():
        pattern = ob.OBSmartsPattern()
        pattern.Init(smarts)
        if pattern.Match(molecule):
            break  # Stop checking SMARTS patterns if a match is found
    else:
        return False  # No match found
    
    # Define the molecular weight range
    min_weight = 110
    max_weight = 200
    
    # Check if the molecular weight is within the specified range
    if not (min_weight <= molecule.GetMolWt() <= max_weight):
        return False
    
    # Define the elemental composition based on the amino acid name
    if AA_name in ("CYS", "MET"):
        elements_in_molecule = {6, 1, 7, 8, 16}  # C, H, N, O, S
    else:
        elements_in_molecule = {6, 1, 7, 8}  # C, H, N, O
    
    # Check if the molecule contains only the allowed elements
    atomic_nums_in_molecule = set(atom.GetAtomicNum() for atom in ob.OBMolAtomIter(molecule))
    if not atomic_nums_in_molecule.issubset(elements_in_molecule):
        return False
    
    return True  # All conditions met

def process_xyz_files2(input_folder, aminoacid_folder, ligand_folder, smarts_pattern, lig_name, AA_name):
    obconv = initialize_obconversion()

    # Ensure output directories exist
    if not os.path.exists(aminoacid_folder):
        # print(f"Creating directory: {aminoacid_folder}")
        os.makedirs(aminoacid_folder, exist_ok=True)
    if not os.path.exists(ligand_folder):
        # print(f"Creating directory: {ligand_folder}")
        os.makedirs(ligand_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".xyz"):
            file_path = os.path.join(input_folder, filename)
            cluster = ob.OBMol()
            success = obconv.ReadFile(cluster, file_path)
            if not success:
                # print(f"Failed to read XYZ file: {file_path}")
                continue

            mols = list(cluster.Separate())
            # print(f"Number of molecules separated: {len(mols)}")
            if len(mols)==2:
                if  sum(is_amino_acid2(mol, smarts_pattern,AA_name) for mol in mols)==1:  #uniqAA-uniq lig (checked)
                    
                    for idx, mol in enumerate(mols):
                        if is_amino_acid2(mol, smarts_pattern,AA_name):
                            aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                            try:
                                obconv.WriteFile(mol, aminoacid_path)
                                # print(f"Written amino acid {idx + 1} to {aminoacid_path}")
                            except Exception as e:
                                print(f"Failed to write amino acid file: {aminoacid_path}. Error: {e}")
                        else:
                            ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                            try:
                                obconv.WriteFile(mol, ligand_path)
                                # print(f"Written ligand {idx + 1} to {ligand_path}")
                            except Exception as e:
                                print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                else:
                    if sum(is_ligand(mol,AA_name) for mol in mols) == 1: #check unique types of atom elements,(checked)
                        for idx, mol in enumerate(mols):
                            if is_ligand(mol,AA_name):
                                ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                try:
                                    obconv.WriteFile(mol, ligand_path)
                                except Exception as e:
                                    print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                            else:
                                aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                try:
                                    obconv.WriteFile(mol, aminoacid_path)
                                except Exception as e:
                                    print(f"Failed to write amino acid file: {aminoacid_path}. Error: {e}")
                    else:                                
                        lig_smarts_pattern="C1NO1"
                        if sum(is_amino_acid(mol, lig_smarts_pattern,AA_name) for mol in mols) == 1:
                            for i, mol in enumerate(mols):
                                if is_amino_acid(mol, lig_smarts_pattern,AA_name):
                                    aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                    try:
                                        obconv.WriteFile(mol, aminoacid_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                                else:
                                    ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                    try:
                                        obconv.WriteFile(mol, ligand_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {ligand_path}. Error: {e}")          
                    
                        else:                                                                                                                                                                               
                            lig_smarts_pattern = get_smiles_from_ligand_name(lig_name)      #check exact smiles_pattern of ligand

                            for i, mol in enumerate(mols):                                                                                                            
                                if is_lig_smiles(mol, lig_smarts_pattern):
                                    ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                    try:
                                        obconv.WriteFile(mol, ligand_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                                else:
                                    aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                    try:
                                        obconv.WriteFile(mol, aminoacid_path)
                                    except Exception as e:
                                        print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")

            else:          
                non_amino_acids = ob.OBMol()
                if sum(is_amino_acid2(mol, smarts_pattern,AA_name) for mol in mols)==1:                   
                    for idx, mol in enumerate(mols):
                        if is_amino_acid2(mol, smarts_pattern,AA_name): # uniqAA- uniq Lig( more than 2 clusters, checked)
                            aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                            try:
                                obconv.WriteFile(mol, aminoacid_path)
                                # print(f"Written amino acid {idx + 1} to {aminoacid_path}")
                            except Exception as e:
                                print(f"Failed to write amino acid file: {aminoacid_path}. Error: {e}")
                        else:                                 
                            non_amino_acids += mol
                        ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                        try:
                            obconv.WriteFile(non_amino_acids, ligand_path)
                        except Exception as e:
                            print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                else:
                        
                    lig_smarts_pattern="C1NO1"
                    non_amino_acids = ob.OBMol()
                    if sum(is_amino_acid(mol, lig_smarts_pattern,AA_name) for mol in mols) == 1:
                        for i, mol in enumerate(mols):
                            if is_amino_acid(mol, lig_smarts_pattern,AA_name):
                                aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                                try:
                                    obconv.WriteFile(mol, aminoacid_path)
                                except Exception as e:
                                    print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                            else:
                                non_amino_acids += mol
                            ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                        
                            try:
                                obconv.WriteFile(non_amino_acids, ligand_path)
                            except Exception as e:
                                print(f"Failed to write ligand file: {ligand_path}. Error: {e}")                       
                                        
                    if sum(is_ligand(mol,AA_name) for mol in mols) == 1: # Accumulate non-amino acids/check unique types of atom elements ( more than 2 clusters, )
                        non_amino_acids = ob.OBMol()
                        for idx, mol in enumerate(mols):
                            if is_ligand(mol,AA_name):
                                ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                try:
                                    obconv.WriteFile(mol, ligand_path)
                                except Exception as e:
                                    print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                            else:
                                non_amino_acids += mol
                        aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                        try:
                            obconv.WriteFile(non_amino_acids, aminoacid_path)
                        except Exception as e:
                            print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                    else:
                            # if sum(count_NO_in_molecule(mol,AA_name) for mol in mols) == 1:

                            # Accumulate non-amino acids via specifying exact ligand smiles code( more than 2 clusters, checked )
                        lig_smarts_pattern = get_smiles_from_ligand_name(lig_name)
                        non_amino_acids = ob.OBMol()
                        molecular_formula = None
                        
                        if lig_smarts_pattern:
                            mol_pattern = Chem.MolFromSmiles(lig_smarts_pattern)
                            molecular_formula = rdMolDescriptors.CalcMolFormula(mol_pattern)
                        
                        for idx, mol in enumerate(mols):
                            # if (is_lig_smiles(mol, lig_smarts_pattern) ) or (molecular_formula == mol.GetFormula()):
                            if (lig_smarts_pattern and is_lig_smiles(mol, lig_smarts_pattern)) or (molecular_formula and molecular_formula == mol.GetFormula()):
    
                                ligand_path = os.path.join(ligand_folder, f"{os.path.splitext(filename)[0]}_ligand.xyz")
                                try:
                                    obconv.WriteFile(mol, ligand_path)
                                except Exception as e:
                                    print(f"Failed to write ligand file: {ligand_path}. Error: {e}")
                            else:
                                non_amino_acids += mol
                            
                        aminoacid_path = os.path.join(aminoacid_folder, f"{os.path.splitext(filename)[0]}_aminoacid.xyz")
                        try:
                            obconv.WriteFile(non_amino_acids, aminoacid_path)
                        except Exception as e:
                            print(f"Failed to write ligand file: {aminoacid_path}. Error: {e}")
                            
                            
                            


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
        
def update_dataset(f, dataset_name, shape, maxshape, dtype='float64'):
    if dataset_name in f:
        del f[dataset_name]  # Delete the existing dataset if it exists
    dataset = f.create_dataset(dataset_name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True)
    return dataset  # Return the newly created dataset


def main():
    parser = argparse.ArgumentParser(description='Process HDF5 file and extract data.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the HDF5 file.')
    parser.add_argument('--input_folder', type=str, default=os.getcwd(), help='Input folder path.')
    parser.add_argument('--aminoacid_folder', type=str, default=os.path.join(os.getcwd(), 'amino'), help='Amino acid folder path.')
    parser.add_argument('--ligand_folder', type=str, default=os.path.join(os.getcwd(), 'ligand'), help='Ligand folder path.')

    args = parser.parse_args()

    file_path = args.file_path
    input_folder = args.input_folder
    aminoacid_folder = args.aminoacid_folder
    ligand_folder = args.ligand_folder

    # Ensure the aminoacid and ligand folders exist
    os.makedirs(aminoacid_folder, exist_ok=True)
    os.makedirs(ligand_folder, exist_ok=True)

    with h5py.File(file_path, 'a') as f:  # Open the file in append mode
        
        for group_name in f.keys():
            group = f[group_name]  # e.g., "007_ARG PRO"
            
            if len(group_name.split()) == 2:
                lig_words, AA_words = group_name.split()
            else:
                print(f"Group name '{group_name}' does not split into two parts. Skipping.")
                continue
            
            if AA_words == "PRO":
                smarts_pattern = initialize_smarts_patterns_PRO()  
            else:
                smarts_pattern = initialize_smarts_patterns()  # Use default pattern initialization
            
            if AA_words == "GLY":
                smarts_pattern = 'C(C(=O)NC)(NC(C)=O)'
        
            coord_complex = read_smiles_from_group(file_path, group_name, 'conformations')
            
            if len(coord_complex) == 0:
                continue
            coord_AA = read_smiles_from_group(file_path, group_name, 'conformations_AA_ang')
            coord_lig = read_smiles_from_group(file_path, group_name, 'conformations_lig_ang')
                    
    
            num_complex = read_smiles_from_group(file_path, group_name, 'atomic_numbers')
            num_AA = read_smiles_from_group(file_path, group_name, 'atomic_numbers_AA')
            num_lig = read_smiles_from_group(file_path, group_name, 'atomic_numbers_lig')    
        
            valid_index = find_valid_index(coord_AA, 0)
            
            if valid_index is None:

                coord_name_AA = update_dataset(group, 'conformations_AA_ang', shape=(coord_complex.shape[0], 0, coord_complex.shape[2]),
                                               maxshape=coord_complex.shape)
                coord_name_lig = update_dataset(group, 'conformations_lig_ang', shape=(coord_complex.shape[0], 0, coord_complex.shape[2]),
                                                maxshape=coord_complex.shape)
                num_name_AA = update_dataset(group, 'atomic_numbers_AA', shape=(coord_complex.shape[1],),
                                             maxshape=(coord_complex.shape[1],), dtype='i2')
                num_name_lig = update_dataset(group, 'atomic_numbers_lig', shape=(coord_complex.shape[1],),
                                              maxshape=(coord_complex.shape[1],), dtype='i2')
                         
                for i in range(coord_complex.shape[0]):
                    
                    print(lig_words, AA_words, i)
                    print(f"No valid index found After {i}")
                    
                    save_xyz_file_complex(coord_complex[i], num_complex, f"{lig_words}_{AA_words}{i}.xyz")
                    
                    amino_path = os.path.join(aminoacid_folder, f"{lig_words}_{AA_words}{i}_aminoacid.xyz")  
                    lig_path = os.path.join(ligand_folder, f"{lig_words}_{AA_words}{i}_ligand.xyz")
                        
                    if AA_words == "PRO":
                        process_xyz_files2(input_folder, aminoacid_folder, ligand_folder, smarts_pattern, lig_words, AA_words)
                    else:
                        process_xyz_files(input_folder, aminoacid_folder, ligand_folder, smarts_pattern, lig_words, AA_words)              
    
                    number_AA, coord_AA_data = assign_coord_number(amino_path)
                    number_lig, coord_lig_data = assign_coord_number(lig_path)
                    
                    file_prefix = f"{lig_words}_{AA_words}{i}"
                    delete_files_if_exists(input_folder, file_prefix)
                        
                    coord_name_AA.resize((coord_complex.shape[0], len(coord_AA_data), coord_complex.shape[2]))
                    coord_name_lig.resize((coord_complex.shape[0], len(coord_lig_data), coord_complex.shape[2]))
                    coord_name_AA[i] = coord_AA_data
                    coord_name_lig[i] = coord_lig_data

                    num_name_AA.resize((len(number_AA),))
                    num_name_AA[...] = number_AA
                    num_name_lig.resize((len(number_lig),))
                    num_name_lig[...] = number_lig
                        
                    print(coord_name_AA.shape)
                    print(coord_name_lig.shape)
                    print(num_name_AA.shape)
                    print(num_name_lig.shape)
                
            else:
             
                for i in range(coord_complex.shape[0]):
                    print(lig_words, AA_words, i)
                
                    save_xyz_file_complex(coord_complex[i], num_complex, f"{lig_words}_{AA_words}{i}.xyz")
                    
                    
                    if group_name == "TD3 HIS":
                        amino_indices = [1, 3, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 28, 29, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 57, 58, 59]
                        ligand_indices = [2, 4, 5, 6, 7, 8, 9, 11, 18, 22, 24, 25, 26, 27, 30, 31, 32, 33, 34, 44, 45, 49, 50, 51, 52, 53, 54, 55, 56]
                        separate_xyz(f"{lig_words}_{AA_words}{i}.xyz", amino_indices, ligand_indices, f"{lig_words}_{AA_words}{i}")
                            
                        number_AA_, coord_AA_ = assign_coord_number(os.path.join(aminoacid_folder, f"{lig_words}_{AA_words}{i}_aminoacid.xyz"))
                        number_lig_, coord_lig_ = assign_coord_number(os.path.join(ligand_folder, f"{lig_words}_{AA_words}{i}_ligand.xyz"))
                            
                        delete_files_if_exists(os.getcwd(), f"{lig_words}_{AA_words}{i}")

                        # Initialize coord_AA and coord_lig if not already initialized
                        if 'coord_AA' not in locals():
                            coord_AA = [None] * coord_complex.shape[0]
                        if 'coord_lig' not in locals():
                            coord_lig = [None] * coord_complex.shape[0]

                        coord_AA[i] = coord_AA_
                        coord_lig[i] = coord_lig_

                        group['conformations_AA_ang'][i] = coord_AA[i]  # Overwrite AA coordinates
                        group['conformations_lig_ang'][i] = coord_lig[i]  # Overwrite ligand coordinates
                        
                    else:
                            
                        if check_len_mols(f"{lig_words}_{AA_words}{i}.xyz") <= 2:
                            os.remove(f"{lig_words}_{AA_words}{i}.xyz")
                    
                        else:
                        
                            valid_index = find_valid_index(coord_AA, 0)
                            print(valid_index, lig_words, AA_words, "VALID INDEX")

                                     
                            if valid_index is not None:
                                save_xyz_file_complex(coord_complex[valid_index], num_complex, f"{lig_words}_{AA_words}{valid_index}.xyz")
                                save_xyz_file(coord_AA[valid_index], num_AA, os.path.join(aminoacid_folder, f"{lig_words}_{AA_words}{valid_index}_aminoacid.xyz"))
                                save_xyz_file(coord_lig[valid_index], num_lig, os.path.join(ligand_folder, f"{lig_words}_{AA_words}{valid_index}_ligand.xyz"))

                                original_atoms, original_coords = read_xyz_file(f"{lig_words}_{AA_words}{valid_index}.xyz")
                                amino_atoms, amino_coords = read_xyz_file(os.path.join(aminoacid_folder, f"{lig_words}_{AA_words}{valid_index}_aminoacid.xyz"))
                                ligand_atoms, ligand_coords = read_xyz_file(os.path.join(ligand_folder, f"{lig_words}_{AA_words}{valid_index}_ligand.xyz"))

                                if valid_index != i:
                                    delete_files_if_exists(os.getcwd(), f"{lig_words}_{AA_words}{valid_index}")                  
                        
                                amino_indices = find_indices(original_atoms, amino_atoms)
                                ligand_indices = find_indices(original_atoms, ligand_atoms)

                                separate_xyz(f"{lig_words}_{AA_words}{i}.xyz", amino_indices, ligand_indices, f"{lig_words}_{AA_words}{i}")

                                number_AA_, coord_AA_ = assign_coord_number(os.path.join(aminoacid_folder, f"{lig_words}_{AA_words}{i}_aminoacid.xyz"))
                                number_lig_, coord_lig_ = assign_coord_number(os.path.join(ligand_folder, f"{lig_words}_{AA_words}{i}_ligand.xyz"))
                        
                                delete_files_if_exists(os.getcwd(), f"{lig_words}_{AA_words}{i}")

                                # Initialize coord_AA and coord_lig if not already initialized
                                if 'coord_AA' not in locals():
                                    coord_AA = [None] * coord_complex.shape[0]
                                if 'coord_lig' not in locals():
                                    coord_lig = [None] * coord_complex.shape[0]

                                coord_AA[i] = coord_AA_
                                coord_lig[i] = coord_lig_

                                group['conformations_AA_ang'][i] = coord_AA[i]  # Overwrite AA coordinates
                                group['conformations_lig_ang'][i] = coord_lig[i]  # Overwrite ligand coordinates

if __name__ == "__main__":
    main()
