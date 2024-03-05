import pandas as pd
import numpy as np
tweet_data = pd.read_csv(r'C:\Users\hs414\OneDrive\Desktop\Major_2024\data\hate\users_hate_glove.content',header=None, delimiter="\t")
all_data = pd.read_csv(r'C:\Users\hs414\OneDrive\Desktop\Major_2024\data\hate\users_hate_all.content',header=None, delimiter="\t")
# tweet_data = tweet_data.rename(columns={0: "user_id", 301: "hate_label"})
tweet_data = tweet_data.rename(columns={0: "user_id", 301: "hate_label"})
all_data = all_data.rename(columns={0: "user_id", 321: "hate_label"})
tweet_data = tweet_data[tweet_data['hate_label'] != "other"]

tweet_data.reset_index(inplace=True)
tweet_data.shape
all_data = all_data[all_data['hate_label'] != "other"]
all_data.reset_index(inplace=True)
all_data.shape
with open(r"C:\Users\hs414\OneDrive\Desktop\Major_2024\data\hate\users.edges", "r") as file:
    lines = file.readlines()
    edges = [tuple(map(int, line.strip().split())) for line in lines]


print("Edges extracted from user.edges file:")
edges[:5]
## Removing edges not in our data set
user_id = tweet_data['user_id']
user_id
valid_nodes = set(user_id)
len(valid_nodes)
cleaned_edges = [(source, target) for source, target in edges if source in valid_nodes and target in valid_nodes]
import networkx as nx

# Create an empty graph
tweet_graph = nx.Graph()

# Add nodes from the cleaned dataset
tweet_graph.add_nodes_from(valid_nodes)

# Add edges from the cleaned edges list
tweet_graph.add_edges_from(cleaned_edges)

# You can now perform various operations on the graph
# For example, you can get the number of nodes and edges
num_nodes = tweet_graph.number_of_nodes()
num_edges = tweet_graph.number_of_edges()

# Print the number of nodes and edges
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)

adj_matrix = nx.adjacency_matrix(tweet_graph, nodelist=user_id)
adj_matrix = adj_matrix.toarray()

# Check the number of nodes in the graph
num_nodes = len(user_id)

# Compare the shape of the adjacency matrix with the number of nodes
print("Shape of adjacency matrix:", adj_matrix.shape)
print("Number of nodes in the graph:", num_nodes)

tweet_data.drop(columns=['index'], inplace=True)
all_data.drop(columns=['index'], inplace=True)
tweet_data_features = tweet_data.drop(columns=['user_id', 'hate_label'])
tweet_data_labels = tweet_data['hate_label']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

scaler = StandardScaler()
scaled_features = scaler.fit_transform(tweet_data_features)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(tweet_data_labels)

X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels_encoded, test_size=0.2, random_state=42)

# Assuming 'tweet_graph' is your networkx graph
# You can convert it to PyTorch Geometric format if necessary

# Convert the graph to PyTorch Geometric format if needed
# Assuming 'tweet_graph' is an adjacency matrix

tweet_graph_tensor = torch.tensor(adj_matrix, dtype=torch.long).t().contiguous()
train_graph, test_graph = train_test_split(tweet_graph_tensor, test_size=0.2, random_state=42)
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # Global pooling
        x = torch.sigmoid(self.fc(x))
        return x
    
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# Initialize the model
input_dim = X_train_tensor.shape[1]
hidden_dim = 64
output_dim = 1
model = GCN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Training the model
test_graph = test_graph.t()
train_graph = train_graph.t()
model.train()
for epoch in range(10):
    for data in train_loader:
        optimizer.zero_grad()
        print("Input data shape:", data[0].shape)
        print("Graph data shape:", train_graph.shape)
        outputs = model(data[0], train_graph)  # Pass features and graph data
        loss = criterion(outputs, data[1].view(-1, 1))
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')










with open(r"C:\Users\hs414\OneDrive\Desktop\Major_2024\data\hate\users.edges", "r") as file:
    lines = file.readlines()
    edges = [tuple(map(int, line.strip().split())) for line in lines]

user_id = tweet_data['user_id']
valid_nodes = set(user_id)
cleaned_edges = [(source, target) for source, target in edges if source in valid_nodes and target in valid_nodes]

# Create networkx graph
tweet_graph = nx.Graph()
tweet_graph.add_nodes_from(valid_nodes)
tweet_graph.add_edges_from(cleaned_edges)

# Convert adjacency matrix to PyTorch tensor
adj_matrix = nx.adjacency_matrix(tweet_graph, nodelist=user_id)
adj_matrix = adj_matrix.toarray()
adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)

# Split data
X_train, X_test, y_train, y_test = train_test_split(tweet_data_features, tweet_data_labels, test_size=0.2, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# Generate edge index
num_nodes = adj_matrix.shape[0]
edge_index = torch.tensor(np.vstack(np.nonzero(adj_matrix)), dtype=torch.long)

# Create Data objects for training and testing
train_data = Data(x=X_train_tensor, edge_index=edge_index, y=y_train_tensor)
test_data = Data(x=X_test_tensor, edge_index=edge_index, y=y_test_tensor)

print("Shape of adjacency matrix:", adj_matrix.shape)
print("Number of unique nodes in the graph:", num_nodes)
print("Edge index shape:", edge_index.shape)