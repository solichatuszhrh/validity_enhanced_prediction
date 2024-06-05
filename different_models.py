## MODEL WITH ONLY LDA600 AS PREDICTOR

#### IMPORT LIBRARIEES ####
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
        mse = self.mse_loss(outputs, targets)
        l1_penalty = sum(param.abs().sum() for param in self.model.parameters())
        loss = mse + self.alpha * l1_penalty
        return loss

lasso_loss = LassoLoss(model, alpha=0.01)
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
train_loss_lasso = []
train_loss = []
epochs = 1000

for epoch in range(epochs):
    model.train()
    
    # Zero the gradient buffers
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    
    # Calculate loss with lasso
    loss_lasso = lasso_loss(outputs, Y_train)
    loss_lasso.backward()
    optimizer.step()
    
    # Calculate loss without lasso
    with torch.no_grad():
        loss = criterion(outputs, Y_train)
    
    # Append the losses to the lists
    train_loss_lasso.append(loss_lasso.item())
    train_loss.append(loss.item())
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss_LASSO: {loss_lasso.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#### MODEL EVALUATION ####
# Model evaluation on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss_lasso = lasso_loss(test_outputs, Y_test)
    test_loss = criterion(test_outputs, Y_test)
    print(f'Test Loss with LASSO: {test_loss_lasso.item():.4f}')
    print(f'Test Loss without LASSO: {test_loss.item():.4f}')

# Plot the training loss over epochs
plt.plot(train_loss_lasso, label='MSE With Lasso')
plt.plot(train_loss, label='MSE Without Lasso')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

lda_only = [test_loss_lasso,test_loss]
lda_only_loss = pd.DataFrame(lda_only).astype("float")


## MODEL WITH MORE PREDICTORS
#### IMPORT LIBRARIES ####
import pandas as pd
import glob
import os
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
        mse = self.mse_loss(outputs, targets)
        l1_penalty = sum(param.abs().sum() for param in self.model.parameters())
        loss = mse + self.alpha * l1_penalty
        return loss

lasso_loss = LassoLoss(model, alpha=0.01)
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
train_loss_lasso = []
train_loss = []
epochs = 1000

for epoch in range(epochs):
    model.train()
    
    # Zero the gradient buffers
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    
    # Calculate loss with lasso
    loss_lasso = lasso_loss(outputs, Y_train)
    loss_lasso.backward()
    optimizer.step()
    
    # Calculate loss without lasso
    with torch.no_grad():
        loss = criterion(outputs, Y_train)
    
    # Append the losses to the lists
    train_loss_lasso.append(loss_lasso.item())
    train_loss.append(loss.item())
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss_LASSO: {loss_lasso.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#### MODEL EVALUATION ####
# Model evaluation on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss_lasso = lasso_loss(test_outputs, Y_test)
    test_loss = criterion(test_outputs, Y_test)
    print(f'Test Loss with LASSO: {test_loss_lasso.item():.4f}')
    print(f'Test Loss without LASSO: {test_loss.item():.4f}')

# Plot the training loss over epochs
plt.plot(train_loss_lasso, label='MSE With Lasso')
plt.plot(train_loss, label='MSE Without Lasso')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

all = [test_loss_lasso,test_loss]
all_loss = pd.DataFrame(all).astype("float")


## SEE THE DIFFERENCE
# Create a table to show loss function
model_loss = pd.concat([lda_only_loss, all_loss], ignore_index=True, axis=1)
model_loss.columns = ['lda_only','all']
model_loss.index = ['loss_with_lasso', 'loss_without_lasso']
print(model_loss)
