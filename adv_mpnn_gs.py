# Imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
import random

# Define the MPNN Model
class MPNNModel(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MPNNModel, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(input_dim, hidden_dim)
        self.lin_out = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        x = F.relu(x)
        x = self.propagate(edge_index, x=x)  # Message passing
        x = self.lin_out(x)
        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.dropout(aggr_out)

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

# Apply adversarial Perturabtions to embeddings
def perturb_embeddings(embeddings, perturb_ratio=0.1): 
    noise = torch.randn_like(embeddings) * perturb_ratio
    return embeddings + noise


# Load embeddings (MPNN or GraphSAGE)
model_type = 'MPNN'  # Change to 'GraphSAGE' for GraphSAGE evaluation
embedding_file = 'graph_embeddings_mpnn.pt' if model_type == 'MPNN' else 'graph_embeddings_graphsage.pt'
embeddings = torch.load(embedding_file, map_location=torch.device('cpu'))

# Apply perturbations
perturbed_embeddings = perturb_embeddings(embeddings)

# Evaluate perturbed embeddings
labels = torch.randint(0, 2, (len(embeddings),))
split_idx = int(0.8 * len(embeddings))
X_train, X_test = embeddings[:split_idx], embeddings[split_idx:]
y_train, y_test = labels[:split_idx], labels[split_idx:]

clf = torch.nn.Linear(X_train.size(1), 2)
optimizer = torch.optim.SGD(clf.parameters(), lr=0.01)

# Differential Privacy
privacy_engine = PrivacyEngine()
clf, optimizer, train_loader = privacy_engine.make_private(
    module=clf,
    optimizer=optimizer,
    data_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32),
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

for epoch in range(20):
    clf.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = clf(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate on test data
clf.eval()
y_pred = clf(X_test).argmax(dim=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Model Type: {model_type}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
