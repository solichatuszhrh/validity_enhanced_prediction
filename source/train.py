import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datetime import datetime
from utils import preprocess_data, dataframe_to_tensor_with_missing, SimpleRegressionModel, LassoLoss, compute_weighted_loss, train_and_evaluate_aug_worst, train_and_evaluate_aug, train_and_evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_values", type=float, default=0.01, help="Regularization strength for Lasso")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of K-Folds for cross-validation")
    args = parser.parse_args()

    alpha = args.alpha
    k_folds = args.k_folds

    # Call K-fold cross-validation
    results = train_and_evaluate_aug_worst(features_tensor, target_tensor, age_groups, k=k_folds, alpha_values=alpha)
    
    for i, (train_loss, val_loss) in enumerate(results):
        print(f"Fold {i+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

if __name__ == "__main__":
    main()

# Step 1: Data preprocessing
# Define the path to the CSV files
path = 'DATA_1'  # Update this to your actual path
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.DataFrame(preprocess_data("DATA_1"))

# Fill NA values before adding age groups
df['age'].fillna(-1, inplace=True)

# Create age_group based on the age column
df.loc[df['age'] <= 19, 'age_groups'] = 'teenage'
df.loc[df['age'].between(20, 24), 'age_groups'] = 'young_adult'
df.loc[df['age'].between(25, 39), 'age_groups'] = 'adult'
df.loc[df['age'].between(40, 64), 'age_groups'] = 'older_adult'
df.loc[df['age'] > 64, 'age_groups'] = 'seniors'

# Convert age groups to categorical codes
age_group_labels = ['teenage', 'young_adult', 'adult', 'older_adult', 'seniors']
df['age_groups'] = pd.Categorical(df['age_groups'], categories=age_group_labels)
age_groups = torch.tensor(df['age_groups'].cat.codes.values, dtype=torch.int64)
df = df.drop(df.loc[:, 'gender':'age_groups'].columns,axis=1)

target_column = ["Factor2","Factor3"]
features_tensor, target_tensor, encoders = dataframe_to_tensor_with_missing(df, target_column)


# Step 2: Add MSE baseline
# Compute the mean of Y_train for both outputs
mean_f2_train = torch.mean(Y_augmented[:, 0])
mean_f3_train = torch.mean(Y_augmented[:, 1])

# Create predictions for the test set
f2_pred = mean_f2_train.repeat(Y_test.shape[0])
f3_pred = mean_f3_train.repeat(Y_test.shape[0])
predictions = torch.stack((f2_pred, f3_pred), dim=1)

# Calculate MSE for the test set
mse_f2 = F.mse_loss(predictions[:, 0], Y_test[:, 0])
mse_f3 = F.mse_loss(predictions[:, 1], Y_test[:, 1])
print(f'Test MSE Baseline (Factor 2): {mse_f2.item():.4f}')
print(f'Test MSE Baseline (Factor 3): {mse_f3.item():.4f}')


# Step 3: Train and evaluate the models
train_and_evaluate(features_tensor, target_tensor)
train_and_evaluate_aug(features_tensor, target_tensor)
train_and_evaluate_aug_worst(features_tensor, target_tensor, age_groups)
