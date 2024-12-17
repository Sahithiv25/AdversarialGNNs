# Imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from opacus import PrivacyEngine

# Define the GCN Model
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

robust_embeddings = torch.load('output_files/robust_graph_embeddings.pt', map_location=torch.device('cpu'))
labels = torch.randint(0, 2, (robust_embeddings.size(0),))

# Split embeddings and labels into train and test sets
split_idx = int(0.8 * len(robust_embeddings))
X_train, X_test = robust_embeddings[:split_idx], robust_embeddings[split_idx:]
y_train, y_test = labels[:split_idx], labels[split_idx:]

clf = LogisticRegression(max_iter=1000)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


model = torch.nn.Linear(X_train.size(1), 2)  # Simple linear classifier
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Train with differential privacy
for epoch in range(10):  # Train for 10 epochs
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} completed")

# On test data
model.eval()
y_pred = model(X_test).argmax(dim=1)
print("Predictions:", y_pred.numpy())
print("True Labels:", y_test.numpy())

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")