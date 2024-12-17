# Imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Define the GraphSAGE Model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# Load the graph data
graphs = torch.load('output_files/graphs.pt', map_location=torch.device('cpu'))

# Split the data into train and test sets
data_size = len(graphs)
train_size = int(0.8 * data_size)
test_size = data_size - train_size
train_data, test_data = torch.utils.data.random_split(graphs, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the model, optimizer
input_dim = graphs[0].x.shape[1]  # Node feature size
hidden_dim = 64
output_dim = 64  # Embedding size

model = GraphSAGEModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        data = data.to(torch.device('cpu'))
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, torch.zeros_like(out))
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# Store Embeddings
model.eval()
embeddings = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(torch.device('cpu'))
        out = model(data.x, data.edge_index)
        embeddings.append(out)

embeddings = torch.cat(embeddings, dim=0)
torch.save(embeddings, 'graph_embeddings_graphsage.pt')
print("GraphSAGE Graph embeddings saved to 'graph_embeddings_graphsage.pt'.")
