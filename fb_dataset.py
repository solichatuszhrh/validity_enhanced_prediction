#### DATA PREPARATION ####
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans
from sklearn.model_selection import train_test_split
import seaborn as sns

# Import train data and test data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df = pd.concat([df_train,df_test])

# Extract factor scores and convert to a PyTorch tensor
factor_scores = df[['Factor2', 'Factor3']].values
factor_scores_tensor = torch.tensor(factor_scores, dtype=torch.float)

# Extract user_ids
user_ids = df['id'].values

# Number of clusters
num_clusters = 5 #based on big five personaluty traits


#### PERFORM CLUSTERING ####
# Perform K-Means clustering
cluster_ids, cluster_centers = kmeans(
    X=factor_scores_tensor, num_clusters=num_clusters, distance='euclidean', device=torch.device('cpu')
)

# Add cluster assignments to the dataframe
df['cluster_id'] = cluster_ids.tolist()

# Plot the clusters using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Factor2', y='Factor3', hue='cluster_id', data=df, palette='viridis', s=100, legend='full')
plt.title('Clusters based on Factor Scores')
plt.xlabel('Factor 2')
plt.ylabel('Factor 3')
plt.legend(title='Cluster')
plt.show()


#### TRAIN A LASSO REGRESSION MODEL ####
# Split the data into training and testing sets (again)
X = df[['Factor2', 'Factor3']].values
y = df['cluster_id'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=351, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.float).reshape(-1, 1)

# Define a simple linear regression model
class LassoRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LassoRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

model = LassoRegression(input_dim=2, output_dim=1)

# Define Lasso loss function
class LassoLoss(nn.Module):
    def __init__(self, l1_lambda):
        super(LassoLoss, self).__init__()
        self.l1_lambda = l1_lambda

    def forward(self, y_pred, y_true, model):
        mse_loss = nn.MSELoss()(y_pred, y_true)
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        return mse_loss + self.l1_lambda * l1_loss

# Hyperparameters
learning_rate = 0.01
num_epochs = 100
l1_lambda = 0.1

criterion = LassoLoss(l1_lambda=l1_lambda)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor, model)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


#### MODEL EVALUATION ####
# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train_tensor)
    y_pred_test = model(X_test_tensor)

    train_loss = criterion(y_pred_train, y_train_tensor, model).item()
    test_loss = criterion(y_pred_test, y_test_tensor, model).item()

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')

