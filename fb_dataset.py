#### IMPORT LIBRARIEES ####
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

#### DATA PREPARATION ####
# Upload additional information
demog = pd.read_csv("demog.csv")
country = pd.read_csv("country.csv")
lda600 = pd.read_csv("LDA600.csv",header=None)
lda600 = lda600.rename(columns={lda600.columns[0]: "userid" })

# Preparing dataset for training
train = pd.read_csv("train.csv")
train = train.rename(columns={"id": "userid","Factor2":"Factor_2","Factor3":"Factor_3"})
with_count = pd.merge(train, country, on="userid")
with_demo = pd.merge(with_count, demog, on="userid")
with_lda = pd.merge(with_demo, lda600, on="userid")
y_train = (with_lda[["Factor_2","Factor_3"]])
x_train = (with_lda.drop(["userid","Factor_2","Factor_3"], axis=1))

# Convert birthday to timestamp and then to float
x_train["birthday"] = pd.to_datetime(x_train["birthday"], errors="coerce")

# Impute missing birthday with mean timestamp
mean_timestamp = x_train["birthday"].dropna().apply(lambda x: x.timestamp()).mean()
x_train["birthday"] = x_train["birthday"].fillna(datetime.fromtimestamp(mean_timestamp))
x_train["birthday_float"] = x_train["birthday"].apply(lambda x: x.timestamp())
x_train = x_train.drop("birthday", axis=1)

# Encode "country" and "locale"
label_encoder_country = LabelEncoder()
label_encoder_locale = LabelEncoder()
x_train["country_encoded"] = label_encoder_country.fit_transform(x_train["country"])
x_train["locale_encoded"] = label_encoder_locale.fit_transform(x_train["locale"])
x_train = x_train.drop("country", axis=1)
x_train = x_train.drop("locale", axis=1)

# Impute NaN values
x_train = x_train.fillna(x_train.mean())

# Preparing dataset for testing
test = pd.read_csv("test.csv")
test = test.rename(columns={"id": "userid","Factor2":"Factor_2","Factor3":"Factor_3"})
test_count = pd.merge(test, country, on="userid")
test_demo = pd.merge(test_count, demog, on="userid")
test_lda = pd.merge(test_demo, lda600, on="userid")
y_test = (test_lda[["Factor_2","Factor_3"]])
x_test = (test_lda.drop(["userid","Factor_2","Factor_3"], axis=1))

# Convert birth_date to timestamp and then to float
x_test["birthday"] = pd.to_datetime(x_test["birthday"], errors="coerce")

# Impute missing birth_date with mean timestamp
mean_timestamp_test = x_test["birthday"].dropna().apply(lambda x: x.timestamp()).mean()
x_test["birthday"] = x_test["birthday"].fillna(datetime.fromtimestamp(mean_timestamp))
x_test["birthday_float"] = x_test["birthday"].apply(lambda x: x.timestamp())
x_test = x_test.drop("birthday", axis=1)

# Encode "country" and "locale"
x_test["country_encoded"] = label_encoder_country.fit_transform(x_test["country"])
x_test["locale_encoded"] = label_encoder_locale.fit_transform(x_test["locale"])
x_test = x_test.drop("country", axis=1)
x_test = x_test.drop("locale", axis=1)

# Impute NaN values 
x_test = x_test.fillna(x_test.mean())

# Convert to PyTorch tensors
X_train = x_train.values
Y_train = y_train.values
X_test = x_test.values
Y_test = y_test.values

X_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
Y_tensor_test = torch.tensor(Y_test, dtype=torch.float32)


#### TRAIN THE MODEL ####
# Build a simple regression model
class SimpleRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

# Initialize the model
input_dim = X_tensor.shape[1]  
output_dim = 2  
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

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    
    # Zero the gradient buffers
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_tensor)
    
    # Calculate loss
    loss = lasso_loss(outputs, Y_tensor)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#### MODEL EVALUATION ####
# Model evaluation on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_tensor_test)
    test_loss = lasso_loss(test_outputs, Y_tensor_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Plot loss function
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
