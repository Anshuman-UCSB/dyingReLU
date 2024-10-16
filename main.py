import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import networkx as nx
import matplotlib.pyplot as plt

# Fixing the random seed
torch.manual_seed(42)


# Define the model
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


model = SimpleNet()
print(model)

# Generate dataset
X = torch.randint(-10, 10, (20, 2)).float()
Y = (X.sum(axis=1) > 0).float().unsqueeze(1)


def train_model():
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

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def visualize_model():
    # Create a graph
    G = nx.DiGraph()

    # Define nodes
    input_nodes = ["x1", "x2"]
    hidden1_nodes = ["h1_1", "h1_2", "h1_3"]
    hidden2_nodes = ["h2_1", "h2_2", "h2_3"]
    output_nodes = ["y"]
    bias_nodes = ["b1", "b2", "b3"]

    # Add nodes to the graph
    G.add_nodes_from(
        input_nodes + hidden1_nodes + hidden2_nodes + output_nodes + bias_nodes
    )

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

    # Position bias nodes
    for i, node in enumerate(bias_nodes):
        pos[node] = (0.5 + i, layer_gap * 2.5)

    # Draw the nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=input_nodes,
        node_color="green",
        node_size=100,
        label="Input Layer",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=hidden1_nodes,
        node_color="red",
        node_size=100,
        label="h1 Layer",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=hidden2_nodes,
        node_color="red",
        node_size=100,
        label="h2 Layer",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=output_nodes,
        node_color="blue",
        node_size=100,
        label="Output Layer",
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=bias_nodes,
        node_color="yellow",
        node_size=100,
        label="Bias Nodes",
    )

    # Extract weights and biases from the model
    weights = {}
    biases = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            layer = name.split(".")[0]
            if layer == "fc1":
                for i, input_node in enumerate(input_nodes):
                    for j, hidden1_node in enumerate(hidden1_nodes):
                        weights[(input_node, hidden1_node)] = param[j, i].item()
            elif layer == "fc2":
                for i, hidden1_node in enumerate(hidden1_nodes):
                    for j, hidden2_node in enumerate(hidden2_nodes):
                        weights[(hidden1_node, hidden2_node)] = param[j, i].item()
            elif layer == "fc3":
                for i, hidden2_node in enumerate(hidden2_nodes):
                    for j, output_node in enumerate(output_nodes):
                        weights[(hidden2_node, output_node)] = param[j, i].item()
        elif "bias" in name:
            layer = name.split(".")[0]
            if layer == "fc1":
                for j, hidden1_node in enumerate(hidden1_nodes):
                    biases[(bias_nodes[0], hidden1_node)] = param[j].item()
            elif layer == "fc2":
                for j, hidden2_node in enumerate(hidden2_nodes):
                    biases[(bias_nodes[1], hidden2_node)] = param[j].item()
            elif layer == "fc3":
                for j, output_node in enumerate(output_nodes):
                    biases[(bias_nodes[2], output_node)] = param[j].item()

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edgelist=weights.keys())
    nx.draw_networkx_edges(G, pos, edgelist=biases.keys(), edge_color="yellow")

    # Add edge labels with weights and biases
    edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in weights.items()}
    bias_labels = {(u, v): f"{b:.2f}" for (u, v), b in biases.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=bias_labels, label_pos=0.4)

    plt.savefig("network.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimpleNet Training and Visualization."
    )
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the model.")

    args = parser.parse_args()

    if args.train:
        train_model()
    if args.visualize:
        visualize_model()

    # Call model on inputs -4 and 5
    inputs = torch.tensor([-4.0, 5.0])
    output = model(inputs)
    print(f"Model output for inputs (-4, 5): {output}")
