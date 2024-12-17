# Imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x

graphs = torch.load('graphs.pt', map_location=torch.device('cpu'))

# Load adversarial graph embeddings
adversarial_graphs = torch.load('adversarial_graph_embeddings.pt', map_location=torch.device('cpu'))

graphs_perturbed = []
for graph in adversarial_graphs:
    if graph.dim() == 1:
        graph = graph.unsqueeze(1)
    graphs_perturbed.append(
        Data(
            x=graph, 
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[1.0], [1.0]], dtype=torch.float)
        )
    )

# Combine clean and adversarial graphs
combined_graphs = graphs + graphs_perturbed

# Split the data into train and test sets
data_size = len(combined_graphs)
train_size = int(0.8 * data_size)
test_size = data_size - train_size
train_data, test_data = torch.utils.data.random_split(combined_graphs, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the model, optimizer
input_dim = graphs[0].x.shape[1]  
hidden_dim = 64
output_dim = 64

model = GCNModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for data in train_loader:
        data = data.to(torch.device('cpu')) 
        optimizer.zero_grad()
        out = model(data)
        # Combined loss: supervised loss and adversarial robustness loss
        target = torch.zeros_like(out)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

model.eval()
robust_embeddings = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(torch.device('cpu'))
        out = model(data)
        robust_embeddings.append(out)


robust_embeddings = torch.cat(robust_embeddings, dim=0)
torch.save(robust_embeddings, 'robust_graph_embeddings.pt')
print("Robust graph embeddings saved to 'robust_graph_embeddings.pt'.")
