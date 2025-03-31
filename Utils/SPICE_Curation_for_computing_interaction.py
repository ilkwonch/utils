import os
import h5py
import numpy as np
import argparse

from utils import (
    initialize_smarts_patterns,
    initialize_smarts_patterns_PRO,
    read_smiles_from_group,
    find_valid_index,
    update_dataset,
    save_xyz_file_complex,
    process_xyz_files,
    process_xyz_files2,
    assign_coord_number,
    delete_files_if_exists,
    check_len_mols,
    save_xyz_file,
    read_xyz_file,
    find_indices,
    separate_xyz,
)


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

    os.makedirs(aminoacid_folder, exist_ok=True)
    os.makedirs(ligand_folder, exist_ok=True)

    with h5py.File(file_path, 'a') as f:  
        
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
                    
                    
