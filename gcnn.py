import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict
from torch_geometric.nn import GCNConv

# Define the Graph Convolutional Neural Network (GCN) model
class GCNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define the Supervised GCNN class
class SupervisedGCNN(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, output_dim, w):
        super(SupervisedGCNN, self).__init__()
        self.gcn = GCNN(input_dim, hidden_dim, output_dim)
        self.w = torch.FloatTensor(w)  # Convert list to torch tensor
        self.xent = nn.CrossEntropyLoss(weight=self.w)

    def forward(self, features, edge_index):
        x = features.weight
        x = self.gcn(x, edge_index)
        return x

    def loss(self, features, edge_index, labels):
        scores = self.forward(features, edge_index)
        return self.xent(scores, labels)

# Function to load dataset and preprocess
def load_data(features, edges, num_features, num_nodes):
    feat_data = np.zeros((num_nodes, num_features))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}

    # Load features
    with open(features) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    # Load edges
    adj_lists = defaultdict(set)
    with open(edges) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            if info[0] in node_map and info[1] in node_map:  # Ensure both nodes are in the dataset
                node1 = node_map[info[0]]
                node2 = node_map[info[1]]
                adj_lists[node1].add(node2)
                adj_lists[node2].add(node1)

    return feat_data, labels, adj_lists

# Function to run training and evaluation
def run_gcnn(model, features, edges, num_features, num_nodes, weights, lr=0.01, batch_size=128, num_epochs=10):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    feat_data, labels, adj_lists = load_data(features, edges, num_features, num_nodes)
    
    # Define features as an embedding layer
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    # Prepare the model
    model = model(num_classes=2, input_dim=num_features, hidden_dim=256, output_dim=2, w=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = model.loss(features, torch.tensor(list(adj_lists.items())), torch.LongTensor(labels.squeeze()))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Perform evaluation here
        pass

if __name__ == "__main__":
    # Initialize and run the GCNN model
    num_classes = 2
    input_dim = 300  # Assuming 300-dimensional input features
    hidden_dim = 256
    output_dim = 2
    weights = [1, 0, 10]  # Example weights
    num_epochs = 10
    lr = 0.01
    batch_size = 128
    weights = [1, 0, 10] 

    model = SupervisedGCNN(num_classes, input_dim, hidden_dim, output_dim, weights)
    run_gcnn(model, features=r"C:\Users\hs414\OneDrive\Desktop\Major_2024\data\hate\users_hate_glove.content", edges=r"C:\Users\hs414\OneDrive\Desktop\Major_2024\data\hate\users.edges", num_features=num_features, 
             num_nodes=num_nodes, weights=weights, lr=lr, batch_size=batch_size, num_epochs=num_epochs)
