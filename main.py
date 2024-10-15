import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Fixing the random seed
torch.manual_seed(42)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)          
        return x

# Define the model
model = SimpleNet()
print(model)


# Generate dataset
X = torch.randint(-10, 10, (10, 2)).float()
Y = (X.sum(axis=1) >= 0).float().unsqueeze(1)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print final model parameters
# print("Final model parameters:")
# for param in model.parameters():
#     print(param)

import matplotlib.pyplot as plt
import networkx as nx

# Define the layer sizes
input_neurons = 2
hidden_neurons = 3
hidden2_neurons = 3
output_neurons = 1

# Create a graph object
G = nx.DiGraph()

# Add nodes to the graph
input_nodes = ['Input' + str(i + 1) for i in range(input_neurons)]
hidden1_nodes = ['h1' + str(i + 1) for i in range(hidden_neurons)]
hidden2_nodes = ['h2' + str(i + 1) for i in range(hidden2_neurons)]
output_nodes = ['Output' + str(i + 1) for i in range(output_neurons)]

all_nodes = input_nodes + hidden1_nodes + hidden2_nodes + output_nodes

for node in all_nodes:
    G.add_node(node)

# Define edges and weights (dummy values)
edges = []

# Input to Hidden edges
for i in input_nodes:
    for h in hidden1_nodes:
        edges.append((i, h))

# Input to Hidden edges
for h1 in hidden1_nodes:
    for h2 in hidden2_nodes:
        edges.append((h1, h2))

# Hidden to Output edges
for h in hidden2_nodes:
    for o in output_nodes:
        edges.append((h, o))

# Add edges to the graph
G.add_edges_from(edges)

# Define positions for better visualization
pos = {}
layer_gap = 5  # Gap between layers

# Position input nodes
for i, node in enumerate(input_nodes):
    pos[node] = (0, i * layer_gap + layer_gap / 2)

# Position hidden nodes
for i, node in enumerate(hidden1_nodes):
    pos[node] = (1, i * layer_gap)
for i, node in enumerate(hidden2_nodes):
    pos[node] = (2, i * layer_gap)

# Position output nodes
for i, node in enumerate(output_nodes):
    pos[node] = (3, i * layer_gap + layer_gap)

# Draw the nodes
nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color="green", node_size=100, label="Input Layer")
nx.draw_networkx_nodes(G, pos, nodelist=hidden1_nodes , node_color="red", node_size=100, label="h1 Layer")
nx.draw_networkx_nodes(G, pos, nodelist=hidden2_nodes , node_color="red", node_size=100, label="h2 Layer")
nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color="blue", node_size=100, label="Output Layer")

# Extract weights from the model
weights = {}
for name, param in model.named_parameters():
    if 'weight' in name:
        layer = name.split('.')[0]
        if layer == 'fc1':
            for i, input_node in enumerate(input_nodes):
                for j, hidden1_node in enumerate(hidden1_nodes):
                    weights[(input_node, hidden1_node)] = param[j, i].item()
        elif layer == 'fc2':
            for i, hidden1_node in enumerate(hidden1_nodes):
                for j, hidden2_node in enumerate(hidden2_nodes):
                    weights[(hidden1_node, hidden2_node)] = param[j, i].item()
        elif layer == 'fc3':
            for i, hidden2_node in enumerate(hidden2_nodes):
                for j, output_node in enumerate(output_nodes):
                    weights[(hidden2_node, output_node)] = param[j, i].item()

# Draw the edges
nx.draw_networkx_edges(G, pos, edgelist=weights.keys())

# Add edge labels with weights
edge_labels = {(u, v): f'{w:.2f}' for (u, v), w in weights.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=.2)

# Show the plot
plt.savefig('graph.png')
