from scipy.stats import gaussian_kde
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

hartree2ev = 27.211386245988
ev2kcalmol = 23.0605

def read_smiles_from_group(file_path, group_name, dataset_name):
    """Read a dataset from a specified group, handle KeyError if dataset not found."""
    with h5py.File(file_path, 'r') as f:
        try:
            return f[group_name][dataset_name][()]
        except KeyError:
            print(f"Dataset '{dataset_name}' not found in group '{group_name}'. Skipping...")
            return None

def compute_E_int(total, aa, lig):
    """Compute interaction energy element-wise, return np.nan for invalid entries."""
    E_int = np.full_like(total, np.nan)
    valid_mask = (~np.isnan(total)) & (~np.isnan(aa)) & (~np.isnan(lig)) & (total != 0) & (aa != 0) & (lig != 0)
    E_int[valid_mask] = (total[valid_mask] - (aa[valid_mask] + lig[valid_mask])) * hartree2ev * ev2kcalmol
    return E_int

def calculate_rmse_r2(x, y):
    """Calculate RMSE and R², ignoring NaN and zero values."""
    valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0) & (y != 0)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) == 0 or len(y_valid) == 0:
        return np.nan, np.nan

    rmse = np.sqrt(mean_squared_error(x_valid, y_valid))
    
    if len(x_valid) > 1 and len(y_valid) > 1:
        model = LinearRegression().fit(x_valid.reshape(-1, 1), y_valid)
        r2 = model.score(x_valid.reshape(-1, 1), y_valid)
    else:
        r2 = np.nan

    return rmse, r2

def calculate_mae(x, y):
    """Calculate MAE, ignoring NaN and zero values.""" 
    valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (x != 0) & (y != 0)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if len(x_valid) == 0 or len(y_valid) == 0:
        return np.nan

    mae = mean_absolute_error(x_valid, y_valid)
    return mae

# File path to the HDF5 file

file_path="AminoAcidLigand_spice_v2_MACE.hdf5"
# file_path = "AminoAcidLigand_v2.1_3.hdf5"


# Initialize lists to store data
b97_aimnet2 = []
b97_mace = []
b97_ref = []

# Open the file and process the datasets group by group
with h5py.File(file_path, 'r') as f:
    for group_name in f.keys():
        # Read datasets with error handling
        AIMNET2_b973c = read_smiles_from_group(file_path, group_name, 'AIMNET2_b973c')
        AIMNET2_b973c_AA = read_smiles_from_group(file_path, group_name, 'AIMNET2_b973c_AA')
        AIMNET2_b973c_lig = read_smiles_from_group(file_path, group_name, 'AIMNET2_b973c_lig')

        ORCA_b973c = read_smiles_from_group(file_path, group_name, 'dft_total_energy_b973c')
        ORCA_b973c_AA = read_smiles_from_group(file_path, group_name, 'dft_total_energy_AA_b973c')
        ORCA_b973c_lig = read_smiles_from_group(file_path, group_name, 'dft_total_energy_lig_b973c')

        MACE_b973c = read_smiles_from_group(file_path, group_name, 'MACE_wb97')
        MACE_b973c_AA = read_smiles_from_group(file_path, group_name, 'MACE_wb97_AA')
        MACE_B973C_lig = read_smiles_from_group(file_path, group_name, 'MACE_wb97_lig')
        
        # Calculate interaction energies
        if AIMNET2_b973c is not None and AIMNET2_b973c_AA is not None and AIMNET2_b973c_lig is not None:
            E_int_AIMNET2 = compute_E_int(AIMNET2_b973c, AIMNET2_b973c_AA, AIMNET2_b973c_lig)
            b97_aimnet2.extend(E_int_AIMNET2.flatten())
            
        if ORCA_b973c is not None and ORCA_b973c_AA is not None and ORCA_b973c_lig is not None:
            E_int_b973c = compute_E_int(ORCA_b973c, ORCA_b973c_AA, ORCA_b973c_lig)
            b97_ref.extend(E_int_b973c.flatten())
        
        if MACE_b973c is not None and MACE_b973c_AA is not None and MACE_B973C_lig is not None:
            E_int_MACE = compute_E_int(MACE_b973c, MACE_b973c_AA, MACE_B973C_lig)
            b97_mace.extend(E_int_MACE.flatten())

# Convert lists to numpy arrays
b97_aimnet2 = np.array(b97_aimnet2)
b97_mace = np.array(b97_mace)
b97_ref = np.array(b97_ref)

# Calculate axis limits
x_min = -200
y_min = -200

x_max = max(np.nanmax(b97_ref), np.nanmax(b97_ref))

# y_min = min(np.nanmin(b97_ref), np.nanmin(b97_ref))

y_max = max(np.nanmax(b97_ref), np.nanmax(b97_ref))

# Calculate RMSE, R², and MAE
rmse_aimnet2 = calculate_rmse_r2(b97_aimnet2, b97_ref)
mae_aimnet2 = calculate_mae(b97_aimnet2, b97_ref)
rmse_mace = calculate_rmse_r2(b97_mace, b97_ref)
mae_mace = calculate_mae(b97_mace, b97_ref)

# Create scatter plots using density coloring
fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

# Plot AIMNET2 vs b97-3c with density coloring
valid_mask_aimnet2 = (~np.isnan(b97_ref)) & (~np.isnan(b97_aimnet2)) & (b97_ref != 0) & (b97_aimnet2 != 0)
x_valid_aimnet2 = b97_ref[valid_mask_aimnet2]
y_valid_aimnet2 = b97_aimnet2[valid_mask_aimnet2]

xy_aimnet2 = np.vstack([x_valid_aimnet2, y_valid_aimnet2])
z_aimnet2 = gaussian_kde(xy_aimnet2)(xy_aimnet2)
idx_aimnet2 = z_aimnet2.argsort()
x_valid_aimnet2, y_valid_aimnet2, z_aimnet2 = x_valid_aimnet2[idx_aimnet2], y_valid_aimnet2[idx_aimnet2], z_aimnet2[idx_aimnet2]

scatter_aimnet2 = axs[0].scatter(x_valid_aimnet2, y_valid_aimnet2, c=z_aimnet2, s=20, cmap='inferno', edgecolors='none')
axs[0].set_xlabel('B97-3c [kcal/mol]', fontsize=14)
axs[0].set_ylabel('AIMNET2 [kcal/mol]', fontsize=14)
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(y_min, y_max)
axs[0].text(0.05, 0.97, f'RMSE: {rmse_aimnet2[0]:.4f}', transform=axs[0].transAxes, fontsize=12, verticalalignment='top')
axs[0].text(0.05, 0.94, f'MAE: {mae_aimnet2:.4f}', transform=axs[0].transAxes, fontsize=12, verticalalignment='top')
axs[0].text(0.05, 0.91, f'R²: {rmse_aimnet2[1]:.4f}', transform=axs[0].transAxes, fontsize=12, verticalalignment='top')

# Plot MACE vs b97-3c with density coloring
valid_mask_mace = (~np.isnan(b97_ref)) & (~np.isnan(b97_mace)) & (b97_ref != 0) & (b97_mace != 0)
x_valid_mace = b97_ref[valid_mask_mace]
y_valid_mace = b97_mace[valid_mask_mace]

xy_mace = np.vstack([x_valid_mace, y_valid_mace])
z_mace = gaussian_kde(xy_mace)(xy_mace)
idx_mace = z_mace.argsort()
x_valid_mace, y_valid_mace, z_mace = x_valid_mace[idx_mace], y_valid_mace[idx_mace], z_mace[idx_mace]

scatter_mace = axs[1].scatter(x_valid_mace, y_valid_mace, c=z_mace, s=20, cmap='inferno', edgecolors='none')



axs[1].set_xlabel('B97-3c [kcal/mol]', fontsize=14)
axs[1].set_ylabel('MACE-OFF23 [kcal/mol]', fontsize=14)
axs[1].set_xlim(x_min, x_max)
axs[1].set_ylim(y_min, y_max)
axs[1].text(0.05, 0.97, f'RMSE: {rmse_mace[0]:.4f}', transform=axs[1].transAxes, fontsize=12, verticalalignment='top')
axs[1].text(0.05, 0.94, f'MAE: {mae_mace:.4f}', transform=axs[1].transAxes, fontsize=12, verticalalignment='top')
axs[1].text(0.05, 0.91, f'R²: {rmse_mace[1]:.4f}', transform=axs[1].transAxes, fontsize=12, verticalalignment='top')


ax2 = axs[1].twinx()
ax2.set_ylim(axs[1].get_ylim())
ax2.tick_params(axis='y', labelcolor='black')


plt.tight_layout()

plt.savefig('RMSE_scatter.png', dpi=300, bbox_inches='tight')
plt.show()
