### LDA ONLY AS PREDICTOR ###
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




### USED ALL PREDICTORS ###
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
