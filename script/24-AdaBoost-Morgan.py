import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import os
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LinearRegression

# Set the current working directory to the folder path
folder_path = '.'

# Get all CSV files in the current directory
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

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

# Custom evaluation metric function: calculate RMSE and R2 simultaneously
def custom_metrics(predt: np.ndarray, dtrain: np.ndarray) -> list:
    y_true = dtrain
    mse = mean_squared_error(y_true, predt)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predt)
    return [('rmse', rmse), ('r2', r2)]

# Iterate through all CSV files
for file_name in csv_files:
    data = pd.read_csv(os.path.join(folder_path, file_name))

    # Extract SMILES and target values
    smiles = data.iloc[:, 1]  # Assume SMILES is in the second column
    targets = data.iloc[:, 2]  # Assume target values are in the third column

    # Calculate Morgan
    Morgan_fingerprints = calculate_Morgan_fingerprints(smiles)

    # Convert fingerprints to DataFrame
    df_Morgan = pd.DataFrame(Morgan_fingerprints, index=smiles.index)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_Morgan, targets, test_size=0.1, random_state=42)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'estimator__max_depth': [3, 5, 7] 
    }

    # Define AdaBoost model with DecisionTreeRegressor as base estimator
    base_estimator = DecisionTreeRegressor(max_depth=3)
    model = AdaBoostRegressor(estimator=base_estimator, random_state=42)

    # Define grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(10, shuffle=True, random_state=42), 
                               scoring=make_scorer(r2_score), refit=True, verbose=3, n_jobs=-1)

    # Train the model
    grid_search.fit(X_train, y_train)

    # Save the best model
    best_model = grid_search.best_estimator_
    model_filename = "24-AdaBoost-Morgan-" + file_name.split('.')[0] + "-model.joblib"
    dump(best_model, model_filename)

    # Save the best parameters
    best_params_name = "24-AdaBoost-Morgan-" + file_name.split('.')[0] + "-best_params.txt"
    with open(best_params_name, 'w') as f:
        for key, value in grid_search.best_params_.items():
            f.write(f"{key}: {value}\n")

    # Output best parameters
    print(f'Best parameters for {file_name}: {grid_search.best_params_}')

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
    results_name = "24-AdaBoost-Morgan-" + file_name.split('.')[0] + "-results.txt"
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
    cv_scores = cross_val_score(best_model, df_Morgan, targets, cv=KFold(10, shuffle=True, random_state=42), scoring='r2')
    cv_scores_name = "24-AdaBoost-Morgan-" + file_name.split('.')[0] + "-cv_scores.txt"
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
    plt.title('24-AdaBoost-Morgan Prediction', fontsize=16)

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
    image_name = "24-AdaBoost-Morgan-" + file_name.split('.')[0] + ".png"
    plt.savefig(image_name, transparent=True, bbox_inches='tight')  # Save the scatter plot as a file with a transparent background
    plt.close()