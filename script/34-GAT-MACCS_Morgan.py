import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from torch_geometric.nn import GATConv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set the working directory to the folder path
folder_path = '.'

# Get all CSV files in the current directory
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Check if CUDA is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define a function to calculate MACCS fingerprints
def calculate_MACCS_fingerprints(smiles):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)  # Generate MACCS fingerprint
            fp_bits = [int(bit) for bit in fp.ToBitString()]  # Convert fingerprint to bit string and then to list
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * 167)  # Fill with zeros if the molecule cannot be loaded
    return fingerprints

# Define a function to calculate Morgan fingerprints
def calculate_morgan_fingerprints(smiles, nBits=2048, radius=2):
    fingerprints = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            fp_bits = list(fp)
            fingerprints.append(fp_bits)
        else:
            fingerprints.append([0] * nBits)  # Fill with zeros if the molecule cannot be loaded
    return fingerprints

# Define a function to convert SMILES to graph data
def mol_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),  # Atomic number
            atom.GetDegree(),  # Degree
            atom.GetFormalCharge(),  # Formal charge
            int(atom.GetChiralTag()),  # Chirality
            atom.GetTotalNumHs(),  # Hydrogen count
            int(atom.GetHybridization())  # Hybridization state
        ])
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge features
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])  # Add reverse edge
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create PyG data object
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([target], dtype=torch.float))
    return data

# Define the FPN model
class FPN(nn.Module):
    def __init__(self, fp_dim, hidden_dim, dropout):
        super(FPN, self).__init__()
        self.fc1 = nn.Linear(fp_dim, hidden_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, fp_list):
        fp_list = torch.tensor(fp_list, dtype=torch.float).to(device)
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out

# Define the GATLayer using GATConv
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, heads=1, concat=True):
        super(GATLayer, self).__init__()
        self.conv = GATConv(in_features, out_features, heads=heads, dropout=dropout, concat=concat)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

# Define the GATEncoder
class GATEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, nheads):
        super(GATEncoder, self).__init__()
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout, heads=1, concat=True) for _ in range(nheads)
        ])
        self.out_att = GATLayer(nhid * nheads, nfeat, dropout, heads=1, concat=False)

    def forward(self, atom_features, edge_index):
        # Apply multi-head attention
        atom_features = torch.cat([att(atom_features, edge_index) for att in self.attentions], dim=1)
        # Output attention layer
        atom_features = self.out_att(atom_features, edge_index)
        return atom_features

# Define the FpgnnModel with combined fingerprints
class FpgnnModel(nn.Module):
    def __init__(self, fp_dim_maccs, fp_dim_morgan, hidden_dim, dropout, nfeat, nhid, nheads):
        super(FpgnnModel, self).__init__()
        self.fpn_maccs = FPN(fp_dim_maccs, hidden_dim, dropout)
        self.fpn_morgan = FPN(fp_dim_morgan, hidden_dim, dropout)
        self.gat = GATEncoder(nfeat, nhid, dropout, nheads)
        self.fc = nn.Linear(hidden_dim * 2 + nfeat, 1)  # Adjusted for combined fingerprints

    def forward(self, smiles):
        fp_maccs = calculate_MACCS_fingerprints(smiles)
        fp_morgan = calculate_morgan_fingerprints(smiles)

        fpn_out_maccs = self.fpn_maccs(fp_maccs)
        fpn_out_morgan = self.fpn_morgan(fp_morgan)

        mols = [mol_to_graph(smile, 0) for smile in smiles]
        batched_x = torch.cat([mol.x for mol in mols], dim=0).to(device)
        
        offset = 0
        batched_edge_index = []
        for mol in mols:
            edge_index = mol.edge_index + offset
            batched_edge_index.append(edge_index)
            offset += mol.num_nodes
        batched_edge_index = torch.cat(batched_edge_index, dim=1).to(device)

        gat_out = self.gat(batched_x, batched_edge_index)
        gat_out = torch.stack([torch.mean(gat_out[offset:offset + mol.num_nodes], dim=0) for offset, mol in enumerate(mols)], dim=0)

        combined_out = torch.cat([fpn_out_maccs, fpn_out_morgan, gat_out], dim=1)
        output = self.fc(combined_out)
        return output

# Iterate through all CSV files
for file_name in csv_files:
    # Load data
    data = pd.read_csv(os.path.join(folder_path, file_name))
    smiles = data.iloc[:, 1].tolist()
    targets = data.iloc[:, 2].tolist()

    # Split data
    train_smiles, val_smiles, train_targets, val_targets = train_test_split(smiles, targets, test_size=0.2, random_state=42)

    # Define model and optimizer
    model = FpgnnModel(
        fp_dim_maccs=167, fp_dim_morgan=2048, 
        hidden_dim=128, dropout=0.5, nfeat=6, nhid=64, nheads=8
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping parameters
    early_stopping_patience = 30  # Number of epochs with no improvement after which training will be stopped
    min_epochs = 50
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_path = f"34-GAT-MACCS_Morgan-{file_name.split('.')[0]}-model.pth"

    # Training loop
    for epoch in range(2000):  # Number of epochs
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients
        output = model(train_smiles)  # Forward pass

        # Ensure target shape matches output shape
        train_targets_tensor = torch.tensor(train_targets, dtype=torch.float).view(-1, 1).to(device)
        loss = criterion(output, train_targets_tensor)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        # Evaluate on validation set
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            val_output = model(val_smiles)  # Forward pass on validation set
            val_targets_tensor = torch.tensor(val_targets, dtype=torch.float).view(-1, 1).to(device)
            val_loss = criterion(val_output, val_targets_tensor)  # Calculate validation loss

            # Calculate R² and RMSE for validation set
            val_preds = val_output.cpu().numpy().flatten()
            val_targets = np.array(val_targets)
            val_r2 = r2_score(val_targets, val_preds)
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

            # Calculate R² and RMSE for training set
            train_preds = output.detach().cpu().numpy().flatten()
            train_targets = np.array(train_targets)
            train_r2 = r2_score(train_targets, train_preds)
            train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))

        # Print metrics
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, '
              f'Val Loss: {val_loss.item():.4f}, '
              f'Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}, '
              f'Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}')

        # Check for early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)  # Save the best model

        # Early stopping condition
        if epoch - best_epoch >= early_stopping_patience and epoch >= min_epochs:
            print(f"Early stopping triggered after epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
            break

    # Training complete
    print(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Final evaluation on validation set with the best model
    with torch.no_grad():
        val_output = model(val_smiles)
        val_targets_tensor = torch.tensor(val_targets, dtype=torch.float).view(-1, 1).to(device)
        val_loss = criterion(val_output, val_targets_tensor)
        val_preds = val_output.cpu().numpy().flatten()
        val_r2 = r2_score(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        # Evaluate on training set
        train_output = model(train_smiles)
        train_preds = train_output.cpu().numpy().flatten()
        train_r2 = r2_score(train_targets, train_preds)
        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))

    print(f"Final Validation Metrics (Best Model): "
          f"Loss: {val_loss.item():.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    print(f"Final Training Metrics (Best Model): "
          f"RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")

    # Save training results
    results_name = f"34-GAT-MACCS_Morgan-{file_name.split('.')[0]}-results.txt"
    results = {
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch + 1
    }

    with open(results_name, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    # Output results to console
    print(f"Final Validation Metrics (Best Model): "
          f"Loss: {val_loss.item():.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    print(f"Final Training Metrics (Best Model): "
          f"RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")

    # Plot scatter plot
    plt.figure(figsize=(8, 8), frameon=False)  # Set aspect ratio to 1:1

    # Plot training set scatter plot
    plt.scatter(train_targets, train_preds, alpha=0.3, edgecolor='black', facecolor='blue', label='Training Set')

    # Plot validation set scatter plot
    plt.scatter(val_targets, val_preds, alpha=0.6, edgecolor='black', facecolor='lightgreen', label='Validation Set')

    # Create linear regression model, set intercept to 0
    model = LinearRegression(fit_intercept=False)
    model.fit(
        np.concatenate([train_targets, val_targets]).reshape(-1, 1),
        np.concatenate([train_preds, val_preds])
    )

    # Get slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    # Plot the fitted line, make it longer
    x_min = min(min(train_targets), min(val_targets))
    x_max = max(max(train_targets), max(val_targets))
    x_values = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_line = slope * x_values + intercept
    plt.plot(x_values, y_line, "r-", label=f'R² = {model.score(np.concatenate([train_targets, val_targets]).reshape(-1, 1), np.concatenate([train_preds, val_preds])):.2f}')

    # Set axis limits
    plt.xlim([x_min, x_max])
    plt.ylim([x_min, x_max])

    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'34-GAT-MACCS_Morgan Prediction for {file_name}', fontsize=16)

    # Remove legend and annotate training and test set RMSE and R2 in the legend position
    plt.legend().remove()

    # Annotate training set RMSE and R2, in blue
    plt.text(0.95, 0.15, f'Train RMSE: {train_rmse:.3f}\nR²: {train_r2:.3f}',
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right',
             fontdict=dict(color='blue', weight='bold'))

    # Annotate validation set RMSE and R2, in green
    plt.text(0.95, 0.05, f'Test RMSE: {val_rmse:.3f}\nR²: {val_r2:.3f}',
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right',
             fontdict=dict(color='green', weight='bold'))

    # Remove grid lines
    plt.grid(False)

    # Save the image
    image_name = f"34-GAT-MACCS_Morgan-{file_name.split('.')[0]}.png"
    plt.savefig(image_name, transparent=True, bbox_inches='tight')  # Save the scatter plot as a file with a transparent background
    plt.close()
