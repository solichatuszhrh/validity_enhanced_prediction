### PREDICTOR ONLY LDA 600 ###
#### IMPORT LIBRARIEES ####
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


#### DATA PREPARATION ####
# Upload additional information
lda600 = pd.read_csv("LDA600.csv",header=None)
lda600 = lda600.rename(columns={lda600.columns[0]: "userid" })

# Preparing dataset for training
train = pd.read_csv("train.csv")
train = train.rename(columns={"id": "userid","Factor2":"Factor_2","Factor3":"Factor_3"})
with_lda = pd.merge(train, lda600, on="userid")
y_train = (with_lda[["Factor_2","Factor_3"]])
x_train = (with_lda.drop(["userid","Factor_2","Factor_3"], axis=1))

# Impute NaN values
x_train = x_train.fillna(x_train.mean())

# Convert to PyTorch tensors
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=12345)
X_train = torch.tensor(X_train.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
Y_test = torch.tensor(Y_test.values, dtype=torch.float32)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


#### BASELINE MODEL ####
# Compute the mean of Y_train for both outputs
mean_f2_train = torch.mean(Y_train[:, 0])
mean_f3_train = torch.mean(Y_train[:, 1])

# Create predictions for the test set
f2_pred = mean_f2_train.repeat(Y_test.shape[0])
f3_pred = mean_f3_train.repeat(Y_test.shape[0])
print(f'Mean Prediction for Factor 2: {mean_f2_train.item():.4f}')
print(f'Mean Prediction for Factor 3: {mean_f3_train.item():.4f}')

# Stack predictions to match the shape of Y_test
predictions = torch.stack((f2_pred, f3_pred), dim=1)

# Calculate MSE for the test set
mse_f2 = F.mse_loss(predictions[:, 0], Y_test[:, 0])
mse_f3 = F.mse_loss(predictions[:, 1], Y_test[:, 1])


#### TRAIN THE MODEL ####
# Build a simple regression model
class SimpleRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = X_train.shape[1]  
output_dim = Y_train.shape[1]   
model = SimpleRegressionModel(input_dim, output_dim)

# Print the model to check its architecture
print(model)

# Customize loss function to lasso regression
class LassoLoss(torch.nn.Module):
    def __init__(self, model, alpha=1.0):
        super(LassoLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, outputs, targets):
        mse_1 = self.mse_loss(outputs[:,0], targets[:,0])
        l1_penalty_1 = sum(param.abs().sum() for param in self.model.parameters())
        loss_1 = mse_1 + self.alpha * l1_penalty_1
        mse_2 = self.mse_loss(outputs[:,1], targets[:,1])
        l1_penalty_2 = sum(param.abs().sum() for param in self.model.parameters())
        loss_2 = mse_2 + self.alpha * l1_penalty_2
        return loss_1, loss_2,

lasso_loss = LassoLoss(model, alpha=0.01)
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
train_loss_lasso_1 = []
train_loss_1 = []
train_loss_lasso_2 = []
train_loss_2 = []
epochs = 1000

for epoch in range(epochs):
    model.train()
    
    # Zero the gradient buffers
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    
    # Calculate loss with lasso
    loss_lasso_1, loss_lasso_2 = lasso_loss(outputs, Y_train)
    loss_lasso_1.backward()
    optimizer.step()
    
    # Calculate loss without lasso
    with torch.no_grad():
        loss_1 = criterion(outputs[:,0], Y_train[:,0])
        loss_2 = criterion(outputs[:,1], Y_train[:,0])
    
    # Append the losses to the lists
    train_loss_lasso_1.append(loss_lasso_1.item())
    train_loss_1.append(loss_1.item())
    train_loss_lasso_2.append(loss_lasso_2.item())
    train_loss_2.append(loss_2.item())
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss_LASSO_1: {loss_lasso_1.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss_1: {loss_1.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss_LASSO_2: {loss_lasso_2.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss_2: {loss_2.item():.4f}')

#### MODEL EVALUATION ####
# Model evaluation on the test set
print(f'Test MSE Baseline (Factor 2): {mse_f2.item():.4f}')
print(f'Test MSE Baseline (Factor 3): {mse_f3.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss_lasso_1, test_loss_lasso_2 = lasso_loss(test_outputs, Y_test)
    test_loss_1 = criterion(test_outputs[:,0], Y_test[:,0])
    test_loss_2 = criterion(test_outputs[:,1], Y_test[:,1])
    print(f'Test MSE with LASSO (Factor 2): {test_loss_lasso_1.item():.4f}')
    print(f'Test MSE with LASSO (Factor 3): {test_loss_lasso_2.item():.4f}')
    print(f'Test MSE without LASSO (Factor 2): {test_loss_1.item():.4f}')
    print(f'Test MSE without LASSO (Factor 3): {test_loss_2.item():.4f}')

# Plot the training loss over epochs
plt.plot(train_loss_lasso_1, label='MSE With Lasso Factor 2')
plt.plot(train_loss_1, label='MSE Without Lasso Factor 2')
plt.plot(train_loss_lasso_2, label='MSE With Lasso Factor 3')
plt.plot(train_loss_2, label='MSE Without Lasso Factor 3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

lda_only = [mse_f2,test_loss_1,test_loss_lasso_1, mse_f3,test_loss_2, test_loss_lasso_2]
lda_only_loss = pd.DataFrame(lda_only).astype("float")
print(lda_only_loss)


### USE ALL PREDICTORS ###
#### IMPORT LIBRARIEES ####
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

#### DATA PREPROCESSING ####
# Define the path to the CSV files
path = 'data'  # Update this to your actual path
csv_files = glob.glob(os.path.join(path, "*.csv"))

def preprocess_data(file_path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                # Rename the first column
                first_column_name = df.columns[0]
                df = df.rename(columns={first_column_name: 'userid'})
                dfs.append(df)
            else:
                print(f"File {file} is empty.")

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Verify all DataFrames are read correctly
    if not dfs:
        print("No DataFrames to merge.")
    else:
        # Assuming 'common_column_name' is the common column used for merging
        common_column = 'userid'  # This should match the renamed first column

        # Merge DataFrames iteratively
        merged_df = dfs[0]
        for df in dfs[1:]:
            if common_column in merged_df.columns and common_column in df.columns:
                merged_df = pd.merge(merged_df, df, on=common_column)
            else:
                print(f"Common column '{common_column}' not found in one of the DataFrames.")
                break

    return merged_df



#### CHANGE DATA TYPE TO TENSOR ####
def dataframe_to_tensor_with_missing(df, target_column):
    # Initialize LabelEncoder for string columns
    label_encoders = {}
    
    # Create a dictionary to store tensor parts
    tensor_parts = {}

    # Exclude target column from features
    features = df.drop(["Factor2","Factor3"], axis=1)

    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            # Fill missing values with mean or any strategy you prefer
            features[column] = features[column].fillna(features[column].mean())
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(features[column].values, dtype=torch.float32)
        
        elif pd.api.types.is_string_dtype(features[column]) or features[column].dtype == 'object':
            # Fill missing values with a placeholder, e.g., 'missing'
            features[column] = features[column].fillna('missing')
            le = LabelEncoder()
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(le.fit_transform(features[column].values), dtype=torch.float32)
                label_encoders[column] = le
        
        elif pd.api.types.is_datetime64_any_dtype(features[column]):
            # Fill missing datetime values with a placeholder or strategy
            features[column] = features[column].fillna(pd.Timestamp('1970-01-01'))
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(features[column].values.astype(np.int64) // 10**9, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unsupported column type: {features[column].dtype} in column {column}")

    # Check if any tensors were created
    if not tensor_parts:
        raise ValueError("No valid columns found for conversion to tensors.")

    # Stack tensors along the second dimension (i.e., columns)
    tensors = torch.stack([tensor_parts[col] for col in features.columns if col in tensor_parts], dim=1)

    # Get target variable tensor
    target_tensor = torch.tensor(df[target_column].values, dtype=torch.float32)

    return tensors, target_tensor, label_encoders


df = pd.DataFrame(preprocess_data("data"))

target_column = ["Factor2","Factor3"]
features_tensor, target_tensor, encoders = dataframe_to_tensor_with_missing(df, target_column)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features_tensor, target_tensor, test_size=0.2, random_state=12345)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


#### BASELINE MODEL ####
# Compute the mean of Y_train for both outputs
mean_f2_train = torch.mean(Y_train[:, 0])
mean_f3_train = torch.mean(Y_train[:, 1])

# Create predictions for the test set
f2_pred = mean_f2_train.repeat(Y_test.shape[0])
f3_pred = mean_f3_train.repeat(Y_test.shape[0])
print(f'Mean Prediction for Factor 2: {mean_f2_train.item():.4f}')
print(f'Mean Prediction for Factor 3: {mean_f3_train.item():.4f}')

# Stack predictions to match the shape of Y_test
predictions = torch.stack((f2_pred, f3_pred), dim=1)

# Calculate MSE for the test set
mse_f2 = F.mse_loss(predictions[:, 0], Y_test[:, 0])
mse_f3 = F.mse_loss(predictions[:, 1], Y_test[:, 1])


#### TRAIN THE MODEL ####
# Build a simple regression model
class SimpleRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = X_train.shape[1]  
output_dim = Y_train.shape[1]   
model = SimpleRegressionModel(input_dim, output_dim)

# Print the model to check its architecture
print(model)

# Customize loss function to lasso regression
class LassoLoss(torch.nn.Module):
    def __init__(self, model, alpha=1.0):
        super(LassoLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, outputs, targets):
        mse_1 = self.mse_loss(outputs[:,0], targets[:,0])
        l1_penalty_1 = sum(param.abs().sum() for param in self.model.parameters())
        loss_1 = mse_1 + self.alpha * l1_penalty_1
        mse_2 = self.mse_loss(outputs[:,1], targets[:,1])
        l1_penalty_2 = sum(param.abs().sum() for param in self.model.parameters())
        loss_2 = mse_2 + self.alpha * l1_penalty_2
        return loss_1, loss_2,

lasso_loss = LassoLoss(model, alpha=0.01)
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
train_loss_lasso_1 = []
train_loss_1 = []
train_loss_lasso_2 = []
train_loss_2 = []
epochs = 1000

for epoch in range(epochs):
    model.train()
    
    # Zero the gradient buffers
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    
    # Calculate loss with lasso
    loss_lasso_1, loss_lasso_2 = lasso_loss(outputs, Y_train)
    loss_lasso_1.backward()
    optimizer.step()
    
    # Calculate loss without lasso
    with torch.no_grad():
        loss_1 = criterion(outputs[:,0], Y_train[:,0])
        loss_2 = criterion(outputs[:,1], Y_train[:,0])
    
    # Append the losses to the lists
    train_loss_lasso_1.append(loss_lasso_1.item())
    train_loss_1.append(loss_1.item())
    train_loss_lasso_2.append(loss_lasso_2.item())
    train_loss_2.append(loss_2.item())
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss_LASSO_1: {loss_lasso_1.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss_1: {loss_1.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss_LASSO_2: {loss_lasso_2.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss_2: {loss_2.item():.4f}')

#### MODEL EVALUATION ####
# Model evaluation on the test set
print(f'Test MSE Baseline (Factor 2): {mse_f2.item():.4f}')
print(f'Test MSE Baseline (Factor 3): {mse_f3.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss_lasso_1, test_loss_lasso_2 = lasso_loss(test_outputs, Y_test)
    test_loss_1 = criterion(test_outputs[:,0], Y_test[:,0])
    test_loss_2 = criterion(test_outputs[:,1], Y_test[:,1])
    print(f'Test MSE with LASSO (Factor 2): {test_loss_lasso_1.item():.4f}')
    print(f'Test MSE with LASSO (Factor 3): {test_loss_lasso_2.item():.4f}')
    print(f'Test MSE without LASSO (Factor 2): {test_loss_1.item():.4f}')
    print(f'Test MSE without LASSO (Factor 3): {test_loss_2.item():.4f}')

# Plot the training loss over epochs
plt.plot(train_loss_lasso_1, label='MSE With Lasso Factor 2')
plt.plot(train_loss_1, label='MSE Without Lasso Factor 2')
plt.plot(train_loss_lasso_2, label='MSE With Lasso Factor 3')
plt.plot(train_loss_2, label='MSE Without Lasso Factor 3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

all = [mse_f2,test_loss_1,test_loss_lasso_1, mse_f3,test_loss_2, test_loss_lasso_2]
all_loss = pd.DataFrame(all).astype("float")
print(all_loss)


### USE K-FOLD ###
#### IMPORT LIBRARIEES ####
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

#### DATA PREPROCESSING ####
# Define the path to the CSV files
path = 'data'  # Update this to your actual path
csv_files = glob.glob(os.path.join(path, "*.csv"))

def preprocess_data(file_path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                # Rename the first column
                first_column_name = df.columns[0]
                df = df.rename(columns={first_column_name: 'userid'})
                dfs.append(df)
            else:
                print(f"File {file} is empty.")

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Verify all DataFrames are read correctly
    if not dfs:
        print("No DataFrames to merge.")
    else:
        # Assuming 'common_column_name' is the common column used for merging
        common_column = 'userid'  # This should match the renamed first column

        # Merge DataFrames iteratively
        merged_df = dfs[0]
        for df in dfs[1:]:
            if common_column in merged_df.columns and common_column in df.columns:
                merged_df = pd.merge(merged_df, df, on=common_column)
            else:
                print(f"Common column '{common_column}' not found in one of the DataFrames.")
                break

    return merged_df



#### CHANGE DATA TYPE TO TENSOR ####
def dataframe_to_tensor_with_missing(df, target_column):
    # Initialize LabelEncoder for string columns
    label_encoders = {}
    
    # Create a dictionary to store tensor parts
    tensor_parts = {}

    # Exclude target column from features
    features = df.drop(["Factor2","Factor3"], axis=1)

    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            # Fill missing values with mean or any strategy you prefer
            features[column] = features[column].fillna(features[column].mean())
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(features[column].values, dtype=torch.float32)
        
        elif pd.api.types.is_string_dtype(features[column]) or features[column].dtype == 'object':
            # Fill missing values with a placeholder, e.g., 'missing'
            features[column] = features[column].fillna('missing')
            le = LabelEncoder()
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(le.fit_transform(features[column].values), dtype=torch.float32)
                label_encoders[column] = le
        
        elif pd.api.types.is_datetime64_any_dtype(features[column]):
            # Fill missing datetime values with a placeholder or strategy
            features[column] = features[column].fillna(pd.Timestamp('1970-01-01'))
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(features[column].values.astype(np.int64) // 10**9, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unsupported column type: {features[column].dtype} in column {column}")

    # Check if any tensors were created
    if not tensor_parts:
        raise ValueError("No valid columns found for conversion to tensors.")

    # Stack tensors along the second dimension (i.e., columns)
    tensors = torch.stack([tensor_parts[col] for col in features.columns if col in tensor_parts], dim=1)

    # Get target variable tensor
    target_tensor = torch.tensor(df[target_column].values, dtype=torch.float32)

    return tensors, target_tensor, label_encoders


df = pd.DataFrame(preprocess_data("data"))

target_column = ["Factor2","Factor3"]
features_tensor, target_tensor, encoders = dataframe_to_tensor_with_missing(df, target_column)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features_tensor, target_tensor, test_size=0.2, random_state=12345)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


#### BASELINE MODEL ####
# Compute the mean of Y_train for both outputs
mean_f2_train = torch.mean(Y_train[:, 0])
mean_f3_train = torch.mean(Y_train[:, 1])

# Create predictions for the test set
f2_pred = mean_f2_train.repeat(Y_test.shape[0])
f3_pred = mean_f3_train.repeat(Y_test.shape[0])
print(f'Mean Prediction for Factor 2: {mean_f2_train.item():.4f}')
print(f'Mean Prediction for Factor 3: {mean_f3_train.item():.4f}')
print(f'Test MSE Baseline (Factor 2): {mse_f2.item():.4f}')
print(f'Test MSE Baseline (Factor 3): {mse_f3.item():.4f}')

# Stack predictions to match the shape of Y_test
predictions = torch.stack((f2_pred, f3_pred), dim=1)

# Calculate MSE for the test set
mse_f2 = F.mse_loss(predictions[:, 0], Y_test[:, 0])
mse_f3 = F.mse_loss(predictions[:, 1], Y_test[:, 1])

# Build a simple regression model
class SimpleRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# Print the model to check its architecture
print(model)

# Customize loss function to lasso regression
class LassoLoss(torch.nn.Module):
    def __init__(self, model, alpha=1.0):
        super(LassoLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, outputs, targets):
        mse_1 = self.mse_loss(outputs[:,0], targets[:,0])
        l1_penalty_1 = sum(param.abs().sum() for param in self.model.parameters())
        loss_1 = mse_1 + self.alpha * l1_penalty_1
        mse_2 = self.mse_loss(outputs[:,1], targets[:,1])
        l1_penalty_2 = sum(param.abs().sum() for param in self.model.parameters())
        loss_2 = mse_2 + self.alpha * l1_penalty_2
        return loss_1, loss_2,

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    Y_train_fold, Y_test_fold = Y_train[train_index], Y_train[test_index]
    
    model = SimpleRegressionModel(X_train.shape[1], Y_train.shape[1])
    lasso_loss = LassoLoss(model, alpha=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_lasso_1 = []
    train_loss_1 = []
    train_loss_lasso_2 = []
    train_loss_2 = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_fold)

        # Calculate loss with lasso
        loss_lasso_1, loss_lasso_2 = lasso_loss(outputs, Y_train_fold)
        loss_lasso_1.backward()
        optimizer.step()

        # Calculate loss without lasso
        loss_1 = criterion(outputs[:, 0], Y_train_fold[:, 0])
        loss_2 = criterion(outputs[:, 1], Y_train_fold[:, 1])
        optimizer.step()
        
        # Append the losses to the lists
        train_loss_lasso_1.append(loss_lasso_1.item())
        train_loss_1.append(loss_1.item())
        train_loss_lasso_2.append(loss_lasso_2.item())
        train_loss_2.append(loss_2.item())
    
        if (epoch+1) % 50 == 0:
            print(f' Epoch [{epoch+1}/{epochs}], Loss (LASSO) (Factor 2): {loss_lasso_1.item():.4f}, Loss (MSE) (Factor 2): {loss_1.item():.4f}')
            print(f'Epoch [{epoch+1}/{epochs}], Loss (LASSO) (Factor 3): {loss_lasso_2.item():.4f}, Loss (MSE) (Factor 3): {loss_2.item():.4f}')

   
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss_lasso_1, test_loss_lasso_2 = lasso_loss(test_outputs, Y_test)
        test_loss_1 = criterion(test_outputs[:, 0], Y_test[:, 0])
        test_loss_2 = criterion(test_outputs[:, 1], Y_test[:, 1])
                
        print(f'Test Loss (MSE) (Factor 2): {test_loss_1.item():.4f}')
        print(f'Test Loss (LASSO) (Factor 3): {test_loss_lasso_1.item():.4f}')
        print(f'Test Loss (MSE) (Factor 2): {test_loss_2.item():.4f}')
        print(f'Test Loss (LASSO) (Factor 3): {test_loss_lasso_2.item():.4f}')


# Plot the training loss over epochs
plt.plot(train_loss_lasso_1, label='MSE With Lasso Factor 2')
plt.plot(train_loss_1, label='MSE Without Lasso Factor 2')
plt.plot(train_loss_lasso_2, label='MSE With Lasso Factor 3')
plt.plot(train_loss_2, label='MSE Without Lasso Factor 3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

all_kfold = [mse_f2,test_loss_1,test_loss_lasso_1, mse_f3,test_loss_2, test_loss_lasso_2]
all_loss_kfold = pd.DataFrame(all_kfold).astype("float")
print(all_loss_kfold)



### USE K-FOLD AND AUGMENT THE DATA ###
#### IMPORT LIBRARIEES ####
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

#### DATA PREPROCESSING ####
# Define the path to the CSV files
path = 'data'  # Update this to your actual path
csv_files = glob.glob(os.path.join(path, "*.csv"))

def preprocess_data(file_path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                # Rename the first column
                first_column_name = df.columns[0]
                df = df.rename(columns={first_column_name: 'userid'})
                dfs.append(df)
            else:
                print(f"File {file} is empty.")

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Verify all DataFrames are read correctly
    if not dfs:
        print("No DataFrames to merge.")
    else:
        # Assuming 'common_column_name' is the common column used for merging
        common_column = 'userid'  # This should match the renamed first column

        # Merge DataFrames iteratively
        merged_df = dfs[0]
        for df in dfs[1:]:
            if common_column in merged_df.columns and common_column in df.columns:
                merged_df = pd.merge(merged_df, df, on=common_column)
            else:
                print(f"Common column '{common_column}' not found in one of the DataFrames.")
                break

    return merged_df



#### CHANGE DATA TYPE TO TENSOR ####
def dataframe_to_tensor_with_missing(df, target_column):
    # Initialize LabelEncoder for string columns
    label_encoders = {}
    
    # Create a dictionary to store tensor parts
    tensor_parts = {}

    # Exclude target column from features
    features = df.drop(["Factor2","Factor3"], axis=1)

    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            # Fill missing values with mean or any strategy you prefer
            features[column] = features[column].fillna(features[column].mean())
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(features[column].values, dtype=torch.float32)
        
        elif pd.api.types.is_string_dtype(features[column]) or features[column].dtype == 'object':
            # Fill missing values with a placeholder, e.g., 'missing'
            features[column] = features[column].fillna('missing')
            le = LabelEncoder()
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(le.fit_transform(features[column].values), dtype=torch.float32)
                label_encoders[column] = le
        
        elif pd.api.types.is_datetime64_any_dtype(features[column]):
            # Fill missing datetime values with a placeholder or strategy
            features[column] = features[column].fillna(pd.Timestamp('1970-01-01'))
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(features[column].values.astype(np.int64) // 10**9, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unsupported column type: {features[column].dtype} in column {column}")

    # Check if any tensors were created
    if not tensor_parts:
        raise ValueError("No valid columns found for conversion to tensors.")

    # Stack tensors along the second dimension (i.e., columns)
    tensors = torch.stack([tensor_parts[col] for col in features.columns if col in tensor_parts], dim=1)

    # Get target variable tensor
    target_tensor = torch.tensor(df[target_column].values, dtype=torch.float32)

    return tensors, target_tensor, label_encoders


df = pd.DataFrame(preprocess_data("data"))

target_column = ["Factor2","Factor3"]
features_tensor, target_tensor, encoders = dataframe_to_tensor_with_missing(df, target_column)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features_tensor, target_tensor, test_size=0.2, random_state=12345)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# Function to augment data by adding Gaussian noise
def augment_data(X, Y, num_augmentations=5, noise_std=0.1):
    augmented_X = []
    augmented_Y = []
    
    for _ in range(num_augmentations):
        noise = torch.normal(mean=0, std=noise_std, size=X.shape)
        augmented_X.append(X + noise)
        augmented_Y.append(Y + noise[:, :Y.shape[1]])  # Apply noise to Y as well
    
    # Stack augmented data with the original data
    augmented_X = torch.cat([X] + augmented_X, dim=0)
    augmented_Y = torch.cat([Y] + augmented_Y, dim=0)
    
    return augmented_X, augmented_Y

# Example data augmentation
num_augmentations = 5  # Number of augmentations per original sample
noise_std = 0.1  # Standard deviation of the noise

augmented_X_train, augmented_Y_train = augment_data(X_train, Y_train, num_augmentations, noise_std)


#### BASELINE MODEL ####
# Compute the mean of Y_train for both outputs
mean_f2_train = torch.mean(augmented_Y_train[:, 0])
mean_f3_train = torch.mean(augmented_Y_train[:, 1])

# Create predictions for the test set
f2_pred = mean_f2_train.repeat(Y_test.shape[0])
f3_pred = mean_f3_train.repeat(Y_test.shape[0])
print(f'Mean Prediction for Factor 2: {mean_f2_train.item():.4f}')
print(f'Mean Prediction for Factor 3: {mean_f3_train.item():.4f}')
print(f'Test MSE Baseline (Factor 2): {mse_f2.item():.4f}')
print(f'Test MSE Baseline (Factor 3): {mse_f3.item():.4f}')

# Stack predictions to match the shape of Y_test
predictions = torch.stack((f2_pred, f3_pred), dim=1)

# Calculate MSE for the test set
mse_f2 = F.mse_loss(predictions[:, 0], Y_test[:, 0])
mse_f3 = F.mse_loss(predictions[:, 1], Y_test[:, 1])

# Build a simple regression model
class SimpleRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# Print the model to check its architecture
print(model)

# Customize loss function to lasso regression
class LassoLoss(torch.nn.Module):
    def __init__(self, model, alpha=1.0):
        super(LassoLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, outputs, targets):
        mse_1 = self.mse_loss(outputs[:,0], targets[:,0])
        l1_penalty_1 = sum(param.abs().sum() for param in self.model.parameters())
        loss_1 = mse_1 + self.alpha * l1_penalty_1
        mse_2 = self.mse_loss(outputs[:,1], targets[:,1])
        l1_penalty_2 = sum(param.abs().sum() for param in self.model.parameters())
        loss_2 = mse_2 + self.alpha * l1_penalty_2
        return loss_1, loss_2,

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(augmented_X_train):
    X_train_fold, X_test_fold = augmented_X_train[train_index], augmented_X_train[test_index]
    Y_train_fold, Y_test_fold = augmented_Y_train[train_index], augmented_Y_train[test_index]
    
    model = SimpleRegressionModel(augmented_X_train.shape[1], augmented_Y_train.shape[1])
    lasso_loss = LassoLoss(model, alpha=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_lasso_1 = []
    train_loss_1 = []
    train_loss_lasso_2 = []
    train_loss_2 = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_fold)

        # Calculate loss with lasso
        loss_lasso_1, loss_lasso_2 = lasso_loss(outputs, Y_train_fold)
        loss_lasso_1.backward()
        optimizer.step()

        # Calculate loss without lasso
        loss_1 = criterion(outputs[:, 0], Y_train_fold[:, 0])
        loss_2 = criterion(outputs[:, 1], Y_train_fold[:, 1])
        optimizer.step()
        
        # Append the losses to the lists
        train_loss_lasso_1.append(loss_lasso_1.item())
        train_loss_1.append(loss_1.item())
        train_loss_lasso_2.append(loss_lasso_2.item())
        train_loss_2.append(loss_2.item())
    
        if (epoch+1) % 50 == 0:
            print(f' Epoch [{epoch+1}/{epochs}], Loss (LASSO) (Factor 2): {loss_lasso_1.item():.4f}, Loss (MSE) (Factor 2): {loss_1.item():.4f}')
            print(f'Epoch [{epoch+1}/{epochs}], Loss (LASSO) (Factor 3): {loss_lasso_2.item():.4f}, Loss (MSE) (Factor 3): {loss_2.item():.4f}')

   
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss_lasso_1, test_loss_lasso_2 = lasso_loss(test_outputs, Y_test)
        test_loss_1 = criterion(test_outputs[:, 0], Y_test[:, 0])
        test_loss_2 = criterion(test_outputs[:, 1], Y_test[:, 1])
                
        print(f'Test Loss (MSE) (Factor 2): {test_loss_1.item():.4f}')
        print(f'Test Loss (LASSO) (Factor 3): {test_loss_lasso_1.item():.4f}')
        print(f'Test Loss (MSE) (Factor 2): {test_loss_2.item():.4f}')
        print(f'Test Loss (LASSO) (Factor 3): {test_loss_lasso_2.item():.4f}')


# Plot the training loss over epochs
plt.plot(train_loss_lasso_1, label='MSE With Lasso Factor 2')
plt.plot(train_loss_1, label='MSE Without Lasso Factor 2')
plt.plot(train_loss_lasso_2, label='MSE With Lasso Factor 3')
plt.plot(train_loss_2, label='MSE Without Lasso Factor 3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

all_kfold_aug = [mse_f2,test_loss_1,test_loss_lasso_1, mse_f3,test_loss_2, test_loss_lasso_2]
all_loss_kfold_aug = pd.DataFrame(all_kfold_aug).astype("float")
print(all_loss_kfold_aug)



### USE K-FOLD, DATA AUGMENTATION, AND WORST-GROUP ACCURACY ###
#### IMPORT LIBRARIEES ####
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

#### DATA PREPROCESSING ####
# Define the path to the CSV files
path = 'data'  # Update this to your actual path
csv_files = glob.glob(os.path.join(path, "*.csv"))

def preprocess_data(file_path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                # Rename the first column
                first_column_name = df.columns[0]
                df = df.rename(columns={first_column_name: 'userid'})
                dfs.append(df)
            else:
                print(f"File {file} is empty.")

        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Verify all DataFrames are read correctly
    if not dfs:
        print("No DataFrames to merge.")
    else:
        # Assuming 'common_column_name' is the common column used for merging
        common_column = 'userid'  # This should match the renamed first column

        # Merge DataFrames iteratively
        merged_df = dfs[0]
        for df in dfs[1:]:
            if common_column in merged_df.columns and common_column in df.columns:
                merged_df = pd.merge(merged_df, df, on=common_column)
            else:
                print(f"Common column '{common_column}' not found in one of the DataFrames.")
                break

    return merged_df

#### CHANGE DATA TYPE TO TENSOR ####
def dataframe_to_tensor_with_missing(df, target_column):
    # Initialize LabelEncoder for string columns
    label_encoders = {}
    
    # Create a dictionary to store tensor parts
    tensor_parts = {}

    # Exclude target column from features
    features = df.drop(["Factor2","Factor3"], axis=1)

    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            # Fill missing values with mean or any strategy you prefer
            features[column] = features[column].fillna(features[column].mean())
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(features[column].values, dtype=torch.float32)
        
        elif pd.api.types.is_string_dtype(features[column]) or features[column].dtype == 'object':
            # Fill missing values with a placeholder, e.g., 'missing'
            features[column] = features[column].fillna('missing')
            le = LabelEncoder()
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(le.fit_transform(features[column].values), dtype=torch.float32)
                label_encoders[column] = le
        
        elif pd.api.types.is_datetime64_any_dtype(features[column]):
            # Fill missing datetime values with a placeholder or strategy
            features[column] = features[column].fillna(pd.Timestamp('1970-01-01'))
            if not features[column].isnull().all():
                tensor_parts[column] = torch.tensor(features[column].values.astype(np.int64) // 10**9, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unsupported column type: {features[column].dtype} in column {column}")

    # Check if any tensors were created
    if not tensor_parts:
        raise ValueError("No valid columns found for conversion to tensors.")

    # Stack tensors along the second dimension (i.e., columns)
    tensors = torch.stack([tensor_parts[col] for col in features.columns if col in tensor_parts], dim=1)

    # Get target variable tensor
    target_tensor = torch.tensor(df[target_column].values, dtype=torch.float32)

    return tensors, target_tensor, label_encoders


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
df = df.drop(['age_groups'], axis=1)

target_column = ["Factor2","Factor3"]
features_tensor, target_tensor, encoders = dataframe_to_tensor_with_missing(df, target_column)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test, age_groups_train, age_groups_test = train_test_split(features_tensor, target_tensor, age_groups, test_size=0.2, random_state=12345)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# Function to augment data by adding Gaussian noise
def augment_data(X, Y, num_augmentations=5, noise_std=0.1):
    augmented_X = []
    augmented_Y = []
    
    for _ in range(num_augmentations):
        noise = torch.normal(mean=0, std=noise_std, size=X.shape)
        augmented_X.append(X + noise)
        augmented_Y.append(Y + noise[:, :Y.shape[1]])  
    
    # Stack augmented data with the original data
    augmented_X = torch.cat([X] + augmented_X, dim=0)
    augmented_Y = torch.cat([Y] + augmented_Y, dim=0)
    
    return augmented_X, augmented_Y

num_augmentations = 5  # Number of augmentations per original sample
noise_std = 0.1  # Standard deviation of the noise
augmented_X_train, augmented_Y_train = augment_data(X_train, Y_train, num_augmentations, noise_std)
age_groups_train_augmented = torch.cat([age_groups_train for _ in range(augmented_X_train.shape[0] // X_train.shape[0])])

#### BASELINE MODEL ####
# Compute the mean of Y_train for both outputs
mean_f2_train = torch.mean(augmented_Y_train[:, 0])
mean_f3_train = torch.mean(augmented_Y_train[:, 1])

# Create predictions for the test set
f2_pred = mean_f2_train.repeat(Y_test.shape[0])
f3_pred = mean_f3_train.repeat(Y_test.shape[0])
print(f'Mean Prediction for Factor 2: {mean_f2_train.item():.4f}')
print(f'Mean Prediction for Factor 3: {mean_f3_train.item():.4f}')
print(f'Test MSE Baseline (Factor 2): {mse_f2.item():.4f}')
print(f'Test MSE Baseline (Factor 3): {mse_f3.item():.4f}')

# Stack predictions to match the shape of Y_test
predictions = torch.stack((f2_pred, f3_pred), dim=1)

# Calculate MSE for the test set
mse_f2 = F.mse_loss(predictions[:, 0], Y_test[:, 0])
mse_f3 = F.mse_loss(predictions[:, 1], Y_test[:, 1])

# Build a simple regression model
class SimpleRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# Print the model to check its architecture
print(model)

# Customize loss function to lasso regression
class LassoLoss(torch.nn.Module):
    def __init__(self, model, alpha=1.0):
        super(LassoLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, outputs, targets):
        mse_1 = self.mse_loss(outputs[:,0], targets[:,0])
        l1_penalty_1 = sum(param.abs().sum() for param in self.model.parameters())
        loss_1 = mse_1 + self.alpha * l1_penalty_1
        mse_2 = self.mse_loss(outputs[:,1], targets[:,1])
        l1_penalty_2 = sum(param.abs().sum() for param in self.model.parameters())
        loss_2 = mse_2 + self.alpha * l1_penalty_2
        return loss_1, loss_2,

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(augmented_X_train):
    X_train_fold, X_test_fold = augmented_X_train[train_index], augmented_X_train[test_index]
    Y_train_fold, Y_test_fold = augmented_Y_train[train_index], augmented_Y_train[test_index]
    age_groups_train_fold, age_groups_test_fold = age_groups_train_augmented[train_index], age_groups_train_augmented[test_index]
    
    model = SimpleRegressionModel(augmented_X_train.shape[1], augmented_Y_train.shape[1])
    lasso_loss = LassoLoss(model, alpha=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_lasso_1 = []
    train_loss_1 = []
    train_loss_lasso_2 = []
    train_loss_2 = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_fold)

        # Calculate loss with lasso
        loss_lasso_1, loss_lasso_2 = lasso_loss(outputs, Y_train_fold)
        loss_lasso_1.backward()
        optimizer.step()

        # Calculate loss without lasso
        loss_1 = criterion(outputs[:, 0], Y_train_fold[:, 0])
        loss_2 = criterion(outputs[:, 1], Y_train_fold[:, 1])
        optimizer.step()
        
        # Append the losses to the lists
        train_loss_lasso_1.append(loss_lasso_1.item())
        train_loss_1.append(loss_1.item())
        train_loss_lasso_2.append(loss_lasso_2.item())
        train_loss_2.append(loss_2.item())
    
        if (epoch+1) % 50 == 0:
            print(f' Epoch [{epoch+1}/{epochs}], Loss (LASSO) (Factor 2): {loss_lasso_1.item():.4f}, Loss (MSE) (Factor 2): {loss_1.item():.4f}')
            print(f'Epoch [{epoch+1}/{epochs}], Loss (LASSO) (Factor 3): {loss_lasso_2.item():.4f}, Loss (MSE) (Factor 3): {loss_2.item():.4f}')

    # Evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss_lasso_1, test_loss_lasso_2 = lasso_loss(test_outputs, Y_test)
        test_loss_1 = criterion(test_outputs[:, 0], Y_test[:, 0])
        test_loss_2 = criterion(test_outputs[:, 1], Y_test[:, 1])

        print(f'Test Loss (MSE) (Factor 2): {test_loss_1.item():.4f}')
        print(f'Test Loss (LASSO) (Factor 2): {test_loss_lasso_1.item():.4f}')
        print(f'Test Loss (MSE) (Factor 3): {test_loss_2.item():.4f}')
        print(f'Test Loss (LASSO) (Factor 3): {test_loss_lasso_2.item():.4f}')

    # Evaluate worst group accuracy
    worst_group_accuracy_mse_f2 = {}
    worst_group_accuracy_mse_f3 = {}
    worst_group_accuracy_lasso_f2 = {}
    worst_group_accuracy_lasso_f3 = {}
    
    model.eval()
    with torch.no_grad():
        for age_group in torch.unique(age_groups_test):
            group_indices = (age_groups_test == age_group).nonzero(as_tuple=True)[0]
            group_X_test = X_test[group_indices]
            group_Y_test = Y_test[group_indices]
    
            group_test_outputs = model(group_X_test)
            group_test_loss_1 = criterion(group_test_outputs[:, 0], group_Y_test[:, 0]).item()
            group_test_loss_2 = criterion(group_test_outputs[:, 1], group_Y_test[:, 1]).item()
            group_test_loss_lasso_1, group_test_loss_lasso_2 = lasso_loss(group_test_outputs, group_Y_test)
            group_label = age_group_labels[age_group]
            worst_group_accuracy_mse_f2[group_label] = group_test_loss_1 
            worst_group_accuracy_mse_f3[group_label] = group_test_loss_1 
            worst_group_accuracy_lasso_f2[group_label] = group_test_loss_lasso_1 
            worst_group_accuracy_lasso_f3[group_label] = group_test_loss_lasso_2 
    
    # Print worst group accuracy
    for group, loss in worst_group_accuracy_mse_f2.items():
        print(f'Age Group: {group}, Test Loss (MSE) (Factor 2): {loss:.4f}')
    for group, loss in worst_group_accuracy_mse_f3.items():
        print(f'Age Group: {group}, Test Loss (MSE) (Factor 3): {loss:.4f}')
    for group, loss in worst_group_accuracy_lasso_f2.items():
        print(f'Age Group: {group}, Test Loss LASSO (MSE) (Factor 2): {loss:.4f}')
    for group, loss in worst_group_accuracy_lasso_f3.items():
        print(f'Age Group: {group}, Test Loss LASSO (MSE) (Factor 3): {loss:.4f}')


# Plot the training loss over epochs
plt.plot(train_loss_lasso_1, label='MSE With Lasso Factor 2')
plt.plot(train_loss_1, label='MSE Without Lasso Factor 2')
plt.plot(train_loss_lasso_2, label='MSE With Lasso Factor 3')
plt.plot(train_loss_2, label='MSE Without Lasso Factor 3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

all_kfold_aug_worst = [mse_f2,test_loss_1,test_loss_lasso_1, mse_f3,test_loss_2, test_loss_lasso_2]
all_loss_kfold_aug_worst = pd.DataFrame(all_kfold_aug_worst).astype("float")
print(all_loss_kfold_aug_worst)



### SHOW THE DIFFERENCE ###
# Create a table to show loss function
model_loss = pd.concat([lda_only_loss, all_loss, all_loss_kfold, all_loss_kfold_aug,all_loss_kfold_aug_worst], ignore_index=True, axis=1)
model_loss.columns = ['lda_only','all','all_kfold','all_kfold_aug','all_kfold_aug_worst']
model_loss.index = ['mse baseline_factor2','loss_without_lasso_factor2', 'loss_with_lasso_factor2','mse baseline_factor3','loss_without_lasso_factor3', 'loss_with_lasso_factor3']
print(model_loss)
