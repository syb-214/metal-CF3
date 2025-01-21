import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, Descriptors
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import os
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from morfeus import BuriedVolume, read_xyz

# Set the current working directory to the folder path
folder_path = '.'

# Get all CSV files in the current directory
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Define a function to calculate MACCS fingerprints
def calculate_MACCS_fingerprints(smiles):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fp_bits = list(fp)
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * 167)  # Fill with 0 if the molecule cannot be loaded
    return fingerprints

# Define a function to calculate Morgan fingerprints
def calculate_Morgan_fingerprints(smiles, radius=2, nBits=2048):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fp_bits = list(fp)
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * nBits)  # Fill with 0 if the molecule cannot be loaded
    return fingerprints

# Define a function to calculate LMD descriptors
def calculate_LMD_descriptors(smiles):
    metal_burial_volumes = []
    metal_covalent_degrees = []
    metal_group_counts = []

    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            metal_atoms = identify_metal_atoms(mol)
            if metal_atoms:
                metal_burial_volume = calculate_burial_volume(mol, metal_atoms[0])
                metal_burial_volumes.append(metal_burial_volume)
                metal_covalent_degree = calculate_metal_covalent_degree(mol, metal_atoms[0])
                metal_covalent_degrees.append(metal_covalent_degree)
                metal_group_count = count_specific_groups(mol, metal_atoms)
                metal_group_counts.append(metal_group_count)
            else:
                metal_burial_volumes.append(0)
                metal_covalent_degrees.append(0)
                metal_group_counts.append(0)
        else:
            metal_burial_volumes.append(0)
            metal_covalent_degrees.append(0)
            metal_group_counts.append(0)

    return metal_burial_volumes, metal_covalent_degrees, metal_group_counts

# Define a function to identify metal atoms
def identify_metal_atoms(mol):
    metal_atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ['Ni', 'Co', 'Pd', 'Ag', 'Fe', 'Cu', 'Pt', 'Rh']:
            metal_atoms.append(atom.GetIdx())
    return metal_atoms

# Define a function to calculate metal covalent degree
def calculate_metal_covalent_degree(mol, metal_atom_idx):
    metal_atom = mol.GetAtomWithIdx(metal_atom_idx)
    covalent_degree = 0
    coordinating_atoms = ['P', 'N', 'C']  # Phosphorus, Nitrogen, and sp2 hybridized Carbon

    for neighbor in metal_atom.GetNeighbors():
        neighbor_symbol = neighbor.GetSymbol()
        bond = mol.GetBondBetweenAtoms(metal_atom_idx, neighbor.GetIdx())
        if bond.GetBondType() in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
            if neighbor_symbol in coordinating_atoms:
                if neighbor_symbol == 'C' and neighbor.GetDegree() == 3:
                    continue  # Skip sp2 hybridized Carbon
                elif neighbor_symbol in ['P', 'N']:
                    continue  # Skip Phosphorus and Nitrogen
            covalent_degree += 1
    return covalent_degree

# Define a function to count specific groups
def count_specific_groups(mol, metal_atoms):
    group_counts = {'CF3': 0, 'CH2F': 0, 'CHF2': 0, 'F': 0}
    for atom in mol.GetAtoms():
        if atom.GetIdx() in metal_atoms:
            neighbors = atom.GetNeighbors()
            for neighbor in neighbors:
                if neighbor.GetSymbol() == 'F':
                    group_counts['F'] += 1
                elif neighbor.GetSymbol() == 'C':
                    if any(n.GetSymbol() == 'F' for n in neighbor.GetNeighbors()):
                        fluorine_count = sum(1 for n in neighbor.GetNeighbors() if n.GetSymbol() == 'F')
                        if fluorine_count == 1:
                            group_counts['CH2F'] += 1
                        elif fluorine_count == 2:
                            group_counts['CHF2'] += 1
                        elif fluorine_count == 3:
                            group_counts['CF3'] += 1
    return sum(group_counts.values())

# Define a function to calculate burial volume
def calculate_burial_volume(mol, metal_atom_idx):
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)  # Generate 2D coordinates if no conformer exists

    conf = mol.GetConformer()
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = [conf.GetAtomPosition(i) for i in range(conf.GetNumAtoms())]
    bv = BuriedVolume(elements, coordinates, metal_atom_idx)
    return 1 - bv.fraction_buried_volume

# Iterate through all CSV files
for file_name in csv_files:
    data = pd.read_csv(os.path.join(folder_path, file_name))

    # Extract SMILES and target values
    smiles = data.iloc[:, 1]  # Assume SMILES is in the second column
    targets = data.iloc[:, 2]  # Assume target values are in the third column

    # Calculate fingerprints
    MACCS_fingerprints = calculate_MACCS_fingerprints(smiles)
    Morgan_fingerprints = calculate_Morgan_fingerprints(smiles)

    # Calculate LMD descriptors
    metal_burial_volumes, metal_covalent_degrees, metal_group_counts = calculate_LMD_descriptors(smiles)

    # Combine fingerprints and LMD descriptors
    combined_features = [maccs + morgan + [bv, cd, gc] for maccs, morgan, bv, cd, gc in zip(
        MACCS_fingerprints, Morgan_fingerprints, metal_burial_volumes, metal_covalent_degrees, metal_group_counts)]

    # Convert combined features to DataFrame
    df_combined = pd.DataFrame(combined_features, index=smiles.index)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_combined, targets, test_size=0.1, random_state=42)

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Define XGBoost model
    model = XGBRegressor(random_state=42)

    # Define grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(10, shuffle=True, random_state=42), 
                               scoring=make_scorer(r2_score), refit=True, verbose=3, n_jobs=-1)

    # Train the model
    grid_search.fit(X_train, y_train)

    # Save the best model
    best_model = grid_search.best_estimator_
    model_filename = "20-XGB-MACCS_Morgan_LMD-" + file_name.split('.')[0] + "-model.joblib"
    dump(best_model, model_filename)

    # Save the best parameters
    best_params_name = "20-XGB-MACCS_Morgan_LMD-" + file_name.split('.')[0] + "-best_params.txt"
    with open(best_params_name, 'w') as f:
        for key, value in grid_search.best_params_.items():
            f.write(f"{key}: {value}\n")

    # Predict and evaluate
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    # Save training results
    results_name = "20-XGB-MACCS_Morgan_LMD-" + file_name.split('.')[0] + "-results.txt"
    results = {
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }
    with open(results_name, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    # Output results
    print(f'File: {file_name}')
    print(f'Training RMSE: {train_rmse}, Training R^2: {train_r2}')
    print(f'Testing RMSE: {test_rmse}, Testing R^2: {test_r2}')

    # Ten-fold cross-validation results
    cv_scores = cross_val_score(best_model, df_combined, targets, cv=KFold(10, shuffle=True, random_state=42), scoring='r2')
    cv_scores_name = "20-XGB-MACCS_Morgan_LMD-" + file_name.split('.')[0] + "-cv_scores.txt"
    with open(cv_scores_name, 'w') as f:
        f.write(f"CV R2 Scores: {cv_scores}\n")
        f.write(f"Mean CV R2 Score: {np.mean(cv_scores)}\n")
        f.write(f"Standard Deviation CV R2 Score: {np.std(cv_scores)}\n")

    # Output ten-fold cross-validation results
    print(f'CV R2 Scores: {cv_scores}')
    print(f'Mean CV R2 Score: {np.mean(cv_scores)}')
    print(f'Standard Deviation CV R2 Score: {np.std(cv_scores)}')

    # Plot scatter plot
    plt.figure(figsize=(8, 8), frameon=False)  # Set aspect ratio to 1:1

    # Plot test set scatter plot
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolor='black', facecolor='lightgreen')

    # Plot training set scatter plot
    plt.scatter(y_train, y_train_pred, alpha=0.3, edgecolor='black', facecolor='blue')  # Increase transparency

    # Create linear regression model, set intercept to 0
    model = LinearRegression(fit_intercept=False)
    model.fit(
        np.concatenate([y_train.values, y_test.values]).reshape(-1, 1),
        np.concatenate([y_train_pred, y_test_pred])
    )

    # Get slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    # Plot the fitted line, make it longer
    x_values = np.linspace(10, 100, 100).reshape(-1, 1)  # Generate points according to the range of actual values
    y_line = slope * x_values + intercept  # Calculate corresponding y values
    plt.plot(x_values, y_line, "r-", label=f'R² = {model.score(np.concatenate([y_train.values, y_test.values]).reshape(-1, 1), np.concatenate([y_train_pred, y_test_pred])):.2f}')

    # Set axis limits
    plt.xlim([10, 100])
    plt.ylim([10, 100])

    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title('20-XGB-MACCS_Morgan_LMD Prediction', fontsize=16)

    # Remove legend and annotate training and test set RMSE and R2 in the legend position
    plt.legend().remove()

    # Annotate training set RMSE and R2, in blue
    plt.text(0.95, 0.15, f'Train RMSE: {train_rmse:.3f}\nR²: {train_r2:.3f}',
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right',
             fontdict=dict(color='blue', weight='bold'))

    # Annotate test set RMSE and R2, in green
    plt.text(0.95, 0.05, f'Test RMSE: {test_rmse:.3f}\nR²: {test_r2:.3f}',
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right',
             fontdict=dict(color='green', weight='bold'))

    # Remove grid lines
    plt.grid(False)

    # Save the image
    image_name = "20-XGB-MACCS_Morgan_LMD-" + file_name.split('.')[0] + ".png"
    plt.savefig(image_name, transparent=True, bbox_inches='tight')  # Save the scatter plot as a file with a transparent background
    plt.close()

