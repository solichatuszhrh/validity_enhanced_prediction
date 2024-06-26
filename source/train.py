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
import random
from utils import preprocess_data, dataframe_to_tensor_with_missing, SimpleRegressionModel, LassoLoss, compute_weighted_loss, train_and_evaluate_aug_worst, train_and_evaluate_aug, train_and_evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_values", type=float, default=0.01, help="Regularization strength for Lasso")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of K-Folds for cross-validation")
    args = parser.parse_args()

    alpha_values = args.alpha_values
    k_folds = args.k_folds

 
    # Step 1: Data preprocessing
    # Define the path to the CSV files
    path = "../validity_enhanced_prediction/data"  
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    df = pd.DataFrame(preprocess_data("data"))
    
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

    X = features_tensor
    Y = target_tensor

    # Set a specific seed
    torch.manual_seed(123)
    
    # Step 2: Add MSE baseline
    # Compute the mean of Y_train for both outputs
    mean_f2_train = torch.mean(Y[:, 0])
    mean_f3_train = torch.mean(Y[:, 1])
    
    # Create predictions for the test set
    f2_pred = mean_f2_train.repeat(Y.shape[0])
    f3_pred = mean_f3_train.repeat(Y.shape[0])
    predictions = torch.stack((f2_pred, f3_pred), dim=1)
    
    # Calculate MSE for the test set
    mse_f2 = F.mse_loss(predictions[:, 0], Y[:, 0])
    mse_f3 = F.mse_loss(predictions[:, 1], Y[:, 1])
    print(f'Test MSE Baseline (Factor 2): {mse_f2.item():.4f}')
    print(f'Test MSE Baseline (Factor 3): {mse_f3.item():.4f}')
    results = {'factor_2_without_lasso': [mse_f2.item()], 'factor_3_without_lasso': [mse_f3.item()],'factor_2_with_lasso': [mse_f2.item()], 'factor_3_with_lasso': [mse_f3.item()]}

    
    # Step 3: Train and evaluate the models
    result_1 = train_and_evaluate_aug_worst(X, Y, age_groups, use_augmentation=False, use_worst_group=False)
    result_2 = train_and_evaluate_aug_worst(X, Y, age_groups, use_augmentation=True, use_worst_group=False)
    result_3 = train_and_evaluate_aug_worst(X, Y, age_groups, use_augmentation=False, use_worst_group=True)
    result_4 = train_and_evaluate_aug_worst(X, Y, age_groups, use_augmentation=True, use_worst_group=True)

    # Step 4: Collect the results
    df0 = pd.DataFrame(results)
    df1 = pd.DataFrame([result_1])
    df2 = pd.DataFrame([result_2])
    df3 = pd.DataFrame([result_3])
    df4 = pd.DataFrame([result_4])
    df = pd.concat([df0,df1,df2,df3,df4], ignore_index=True, axis=0)
    #df.columns = ['Factor 2 without Lasso','Factor 3 without Lasso','Factor 2 with Lasso','Factor 3 with Lasso']
    df.index = ['MSE baseline','Model with K-fold', 'Add data augmentation','Add worst-group accuracy','Add data augmentation and worst-group accuracy']

    print(df)
    

if __name__ == "__main__":
    main()
