# Imports
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from torch_geometric.data import Data

# Function to convert SMILES to graph
def smiles_to_graph(smiles):
    if not isinstance(smiles, str):
        raise ValueError(f"SMILES must be a string. Invalid input: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Node features: atomic numbers
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)

    # Edge indices and features: bonds
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append((start, end))
        edge_indices.append((end, start))
        bond_type = bond.GetBondType()
        edge_features.append(bond_type)
        edge_features.append(bond_type)

    # Convert edge indices to PyTorch tensor
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Convert bond types to features
    bond_type_map = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2,
                     Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4}
    edge_attr = torch.tensor([bond_type_map[bond] for bond in edge_features], dtype=torch.float).view(-1, 1)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# Import Dataset
data = pd.read_csv('final_dataset.csv')
df = pd.DataFrame(data)

# Filter out invalid or NaN SMILES
df = df.dropna(subset=['canonical_smiles'])
df = df[df['canonical_smiles'].apply(lambda x: isinstance(x, str))]

# Convert each SMILES to graph
graphs = []

for smiles in df['canonical_smiles']:
    try:
        graph = smiles_to_graph(smiles)
        graphs.append(graph)
    except ValueError as e:
        print(e)

# Display graph's details
if graphs:
    print("Example Graph from canonical_smiles:")
    print("Node features:", graphs[0].x)
    print("Edge indices:", graphs[0].edge_index)
    print("Edge attributes:", graphs[0].edge_attr)

# Save graphs for further usage
torch.save(graphs, 'graphs.pt')
