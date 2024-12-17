# Imports
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def evaluate_model(embedding_file, model_name):

    embeddings = torch.load(embedding_file, map_location=torch.device('cpu'))
    labels = torch.randint(0, 2, (embeddings.size(0),))

    # Split embeddings and labels into train and test sets
    split_idx = int(0.8 * len(embeddings))
    X_train, X_test = embeddings[:split_idx], embeddings[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train.numpy(), y_train.numpy())

    y_pred = clf.predict(X_test.numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Evaluate MPNN Embeddings
evaluate_model('graph_embeddings_mpnn.pt', 'MPNN')

# Evaluate GraphSAGE Embeddings
evaluate_model('graph_embeddings_graphsage.pt', 'GraphSAGE')