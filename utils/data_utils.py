import random
import torch

from torch_geometric.datasets import Planetoid, Yelp, WikipediaNetwork

import numpy as np
from torch_geometric.utils import to_networkx
import csv
import copy
from torch_geometric.data import Data
import os

from torch_geometric.utils import to_undirected

def load_dataset(dataset_name):
    if dataset_name.lower() == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
    elif dataset_name.lower() == 'yelp':
        dataset = Yelp(root='data/yelp')
    elif dataset_name.lower() == 'pubmed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed') 
    elif dataset_name.lower() == 'citeseer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
    elif dataset_name.lower() == 'chameleon':
        dataset = data/chameleon    
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    data = dataset[0]
    print(f"Loaded {dataset_name} dataset with {data.num_nodes} nodes and {data.num_edges} edges.")
    return data

# Split train-test using snowball sampling
def snowball_sampling(data, initial_nodes, max_nodes, max_neighbors):
    """
    Perform snowball sampling on the graph to return train and test subgraphs in the form of PyTorch Geometric Data objects.
    
    Parameters:
    - data: PyTorch Geometric Data object containing the original graph.
    - initial_nodes: List of initial nodes to start the sampling process.
    - max_nodes: Maximum number of nodes to sample.
    - max_neighbors: Maximum number of neighbors to sample at each step.

    Returns:
    - train_data: A PyTorch Geometric Data object containing the sampled train subgraph.
    - test_data: A PyTorch Geometric Data object containing the test subgraph (remaining nodes and edges).
    - train_to_orig: A dictionary mapping node indices in the train subgraph to the original node indices.
    - test_to_orig: A dictionary mapping node indices in the test subgraph to the original node indices.
    """
    # Convert to NetworkX for easy neighbor processing
    G = to_networkx(data, to_undirected=True)

    # Initialize sets for nodes and edges
    sampled_nodes = set(initial_nodes)
    sampled_edges = set()

    # Initialize the frontier with the initial nodes
    frontier = list(initial_nodes)
    
    while len(sampled_nodes) < max_nodes and frontier:
        # Get a random node from the frontier
        current_node = frontier.pop(0)
        neighbors = list(G.neighbors(current_node))

        # Limit the number of neighbors sampled to max_neighbors
        random.shuffle(neighbors)
        neighbors = neighbors[:max_neighbors]

        # Add the neighbors to the sampled nodes and edges
        for neighbor in neighbors:
            if neighbor not in sampled_nodes:
                sampled_nodes.add(neighbor)
                frontier.append(neighbor)

            # Add the edge (current_node, neighbor)
            sampled_edges.add((current_node, neighbor))
            sampled_edges.add((neighbor, current_node))

        # Stop if the number of sampled nodes reaches max_nodes
        if len(sampled_nodes) >= max_nodes:
            break

    # Convert sampled nodes and edges back to tensors for PyTorch Geometric
    sampled_nodes = list(sampled_nodes)
    edge_index_train = torch.tensor(list(sampled_edges), dtype=torch.long).t().contiguous()

    # Extract node features and labels for the sampled nodes
    sampled_node_features = data.x[sampled_nodes]
    sampled_labels = data.y[sampled_nodes]

    # Map sampled nodes to new indices (0, 1, 2, ...) for the subgraph
    orig_to_train_idx = {orig_idx: i for i, orig_idx in enumerate(sampled_nodes)}
    train_edge_index = torch.tensor(
        [[orig_to_train_idx[src], orig_to_train_idx[dst]] for src, dst in sampled_edges],
        dtype=torch.long
    ).t().contiguous()

    # Create the train data object
    train_data = Data(x=sampled_node_features, edge_index=train_edge_index, y=sampled_labels)
    train_to_orig = {v: k for k, v in orig_to_train_idx.items()}

    # Get the remaining nodes and edges for the test data
    all_nodes = set(range(data.num_nodes))
    test_nodes = list(all_nodes - set(sampled_nodes))
    test_edges = [(u, v) for u, v in G.edges() if u in test_nodes and v in test_nodes]

    # Extract node features and labels for the test nodes
    test_node_features = data.x[test_nodes]
    test_labels = data.y[test_nodes]

    # Map test nodes to new indices for the test subgraph
    orig_to_test_idx = {orig_idx: i for i, orig_idx in enumerate(test_nodes)}
    test_edge_index = torch.tensor(
        [[orig_to_test_idx[src], orig_to_test_idx[dst]] for src, dst in test_edges],
        dtype=torch.long
    ).t().contiguous()

    # Create the test data object
    test_data = Data(x=test_node_features, edge_index=test_edge_index, y=test_labels)
    test_to_orig = {v: k for k, v in orig_to_test_idx.items()}

    print(f'Number of nodes in the train graph: {train_data.x.size(0)}')
    print(f'Number of edges in the train graph: {train_data.edge_index.size(1)}')
    print(f'Number of nodes in the test graph: {test_data.x.size(0)}')
    print(f'Number of edges in the test graph: {test_data.edge_index.size(1)}')
    return train_data, test_data, train_to_orig, test_to_orig


# ---------------------------------------------------------------------------
# Snowball sampling with sampled nodes and edges through which they were discovered
# ---------------------------------------------------------------------------
def snowball_sampling_with_train_test(
        data: Data,
        initial_nodes,
        train_ratio: float,
        max_neighbors: int,
        undirected: bool = True,
        rng=random.Random()):
    """
    Parameters
    ----------
    data           : PyG Data (edge_index can already be directed or undirected)
    initial_nodes  : list[int]      seed nodes
    train_ratio    : 0.0–1.0        |V_train| / |V|
    max_neighbors  : int            fan-out per frontier node
    undirected     : bool           force symmetric edge_index first
    rng            : random.Random  (for reproducibility)

    Returns
    -------
    Data with attributes: train_mask, test_mask,
                          train_edge_mask, test_edge_mask
    """
    # ------------------------------------------------------------
    # 0. set-up
    # ------------------------------------------------------------
    if undirected:
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    else:
        edge_index = data.edge_index

    num_nodes   = data.num_nodes
    target_size = int(train_ratio * num_nodes)

    # build adjacency list (Python lists are cheap for irregular fan-outs)
    row, col = edge_index
    neigh_lists = [[] for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        neigh_lists[u].append(v)          # already contains both directions

    # ------------------------------------------------------------
    # 1. snowball growth
    # ------------------------------------------------------------
    sampled_nodes  = set(initial_nodes)
    sampled_edges  = set()                # store (u,v) with BOTH orientations
    frontier       = list(initial_nodes)

    while frontier and len(sampled_nodes) < target_size:
        u = frontier.pop(0)
        nbrs = neigh_lists[u]
        if not nbrs:                      # isolated node
            continue
        
        # print(f'Node {u} has {len(nbrs)} neighbors, sampled {len(sampled_nodes)} nodes so far')
        # print(f'max_neighbors: {max_neighbors}, target_size: {target_size}')
        rng.shuffle(nbrs)
        for v in nbrs[:max_neighbors]:
            # add node
            if v not in sampled_nodes and len(sampled_nodes) < target_size:
                sampled_nodes.add(v)
                frontier.append(v)
            
            # add edges in both directions
            sampled_edges.add((u, v))
            sampled_edges.add((v, u))
            # if v in sampled_nodes:
            #     print(f'{v} is in sampled nodes, adding edge ({u}, {v})')

        # early stop once we hit the quota
        if len(sampled_nodes) >= target_size:
            break

    sampled_nodes = list(sampled_nodes)
    train_edges   = list(sampled_edges)

    # ------------------------------------------------------------
    # 2. build test split
    # ------------------------------------------------------------
    train_set = set(sampled_nodes)
    test_nodes = [n for n in range(num_nodes) if n not in train_set]

    test_edges = []
    for u in test_nodes:
        for v in neigh_lists[u]:
            if v in test_nodes:           # both endpoints in test partition
                test_edges.extend([(u, v)])

    # ------------------------------------------------------------
    # 3. assemble final Data object
    # ------------------------------------------------------------
    combined_edges = train_edges + test_edges
    edge_index_out = torch.tensor(combined_edges, dtype=torch.long).t().contiguous()

    # node masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[sampled_nodes] = True
    test_mask = ~train_mask

    # edge masks (vectorised)
    train_edge_mask = torch.zeros(edge_index_out.size(1), dtype=torch.bool)
    train_edge_mask[:len(train_edges)] = True
    test_edge_mask = ~train_edge_mask

    print(f'Train graph: {train_mask.sum().item()} nodes, {train_edge_mask.sum().item()} edges')
    print(f'Test graph: {test_mask.sum().item()} nodes, {test_edge_mask.sum().item()} edges')

    return Data(
        x=data.x,  y=data.y,
        edge_index=edge_index_out,
        train_mask=train_mask,
        test_mask=test_mask,
        train_edge_mask=train_edge_mask,
        test_edge_mask=test_edge_mask
    )


# ---------------------------------------------------------------------------
# Snowball sampling with sampled nodes and all edges among the train nodes
# ---------------------------------------------------------------------------
def snowball_sampling_with_train_test_induced(
        data: Data,
        initial_nodes,
        train_ratio: float,
        max_neighbors: int,
        undirected: bool = True,
        rng=random.Random()):
    """
    Parameters
    ----------
    data           : PyG Data (edge_index can already be directed or undirected)
    initial_nodes  : list[int]      seed nodes
    train_ratio    : 0.0–1.0        |V_train| / |V|
    max_neighbors  : int            fan-out per frontier node
    undirected     : bool           force symmetric edge_index first
    rng            : random.Random  (for reproducibility)

    Returns
    -------
    Data with attributes: train_mask, test_mask,
                          train_edge_mask, test_edge_mask
    """
    # ------------------------------------------------------------
    # 0. set-up
    # ------------------------------------------------------------
    if undirected:
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    else:
        edge_index = data.edge_index

    num_nodes   = data.num_nodes
    target_size = int(train_ratio * num_nodes)

    # build adjacency list (Python lists are cheap for irregular fan-outs)
    row, col = edge_index
    neigh_lists = [[] for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        neigh_lists[u].append(v)          # already contains both directions

    # ------------------------------------------------------------
    # 1. snowball growth
    # ------------------------------------------------------------
    sampled_nodes  = set(initial_nodes)
    
    frontier       = list(initial_nodes)

    while frontier and len(sampled_nodes) < target_size:
        u = frontier.pop(0)
        nbrs = neigh_lists[u]
        if not nbrs:                      # isolated node
            continue
        
        # print(f'Node {u} has {len(nbrs)} neighbors, sampled {len(sampled_nodes)} nodes so far')
        # print(f'max_neighbors: {max_neighbors}, target_size: {target_size}')
        rng.shuffle(nbrs)
        for v in nbrs[:max_neighbors]:
            # add node
            if v not in sampled_nodes and len(sampled_nodes) < target_size:
                sampled_nodes.add(v)
                frontier.append(v)
            
            # # add edges in both directions
            # sampled_edges.add((u, v))
            # sampled_edges.add((v, u))
            # # if v in sampled_nodes:
            # #     print(f'{v} is in sampled nodes, adding edge ({u}, {v})')

        # early stop once we hit the quota
        if len(sampled_nodes) >= target_size:
            break

    sampled_nodes = list(sampled_nodes)
    train_edges=[]

    for u in sampled_nodes:
        for v in neigh_lists[u]:
            if v in sampled_nodes:           # both endpoints in train partition
                train_edges.extend([(u, v)])

    # ------------------------------------------------------------
    # 2. build test split
    # ------------------------------------------------------------
    train_set = set(sampled_nodes)
    test_nodes = [n for n in range(num_nodes) if n not in train_set]

    test_edges = []
    for u in test_nodes:
        for v in neigh_lists[u]:
            if v in test_nodes:           # both endpoints in test partition
                test_edges.extend([(u, v)])

    # ------------------------------------------------------------
    # 3. assemble final Data object
    # ------------------------------------------------------------
    combined_edges = train_edges + test_edges
    edge_index_out = torch.tensor(combined_edges, dtype=torch.long).t().contiguous()

    # node masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[sampled_nodes] = True
    test_mask = ~train_mask

    # edge masks (vectorised)
    train_edge_mask = torch.zeros(edge_index_out.size(1), dtype=torch.bool)
    train_edge_mask[:len(train_edges)] = True
    test_edge_mask = ~train_edge_mask

    print(f'Train graph: {train_mask.sum().item()} nodes, {train_edge_mask.sum().item()} edges')
    print(f'Test graph: {test_mask.sum().item()} nodes, {test_edge_mask.sum().item()} edges')

    return Data(
        x=data.x,  y=data.y,
        edge_index=edge_index_out,
        train_mask=train_mask,
        test_mask=test_mask,
        train_edge_mask=train_edge_mask,
        test_edge_mask=test_edge_mask
    )

def select_random_nodes_from_each_class(data, num_samples_per_class,rng=random.Random()):
    selected_nodes = []
    num_classes = int(data.y.max().item() + 1)
    for c in range(num_classes):
        class_nodes = (data.y == c).nonzero(as_tuple=True)[0]
        selected_nodes += rng.sample(class_nodes.tolist(), num_samples_per_class)
    return selected_nodes




# First, reverse the mapping dictionary to map from test subgraph indices back to original indices
def extract_original_edges(index_mapping, edges):
    
    # print(" Index", index_mapping)
    # print("Original Edges", edges)
    for i in range(edges.size(0)):
        for j in range(edges.size(1)):
            edges[i, j] = index_mapping[edges[i, j].item()]
    return edges







def sample_negative_edges(graph):
    """
    Sample an equal number of negative test edges from the train graph.
    
    Parameters:
    graph (networkx.Graph): The input graph.

    
    Returns:
    list: List of negative edges.
    """
    negative_edges = set()
    all_nodes = list(graph.nodes())
    train_edges = graph.edges()
    num_train_edges = len(train_edges)
    
    while len(negative_edges) < num_train_edges:
        node1 = random.choice(all_nodes)
        node2 = random.choice(all_nodes)
        if node1 != node2 and not graph.has_edge(node1, node2):
            negative_edges.add((node1, node2))
    
    return list(negative_edges)

def write_edges_to_file(subgraph, filename="train_edges.csv"):
    # Open a file to write the edges
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["source", "target"])
        
        # Write each edge in the subgraph to the file
        for edge in subgraph.edges():
            writer.writerow(edge)


def create_inductive_random_splits(data, train_ratio):
    num_nodes = data.num_nodes
        
    # Calculate the number of nodes for training
    num_train = int(train_ratio * num_nodes)
    # Generate a random permutation of all node indices
    all_indices = torch.randperm(num_nodes)
        
    # Split the indices into train and test sets
    train_idx = all_indices[:num_train]
    test_idx = all_indices[num_train:]
        
    # Initialize lists to store edges for train and test graphs
    train_edges = []
    test_edges = []

    # Manually extract edges for the train and test subgraphs
    for edge in data.edge_index.t().tolist():
        if edge[0] in train_idx and edge[1] in train_idx:
            train_edges.append(tuple(edge))
        elif edge[0] in test_idx and edge[1] in test_idx:
            test_edges.append(tuple(edge))
    
    # Combine train and test edges into a single edge list
    combined_edges = train_edges + test_edges
    edge_index = torch.tensor(combined_edges, dtype=torch.long).t().contiguous()

    # Create node masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True

    test_mask = ~train_mask

    # Convert train_edges to a set for fast lookup
    train_edges_set = set(train_edges)  # O(1) lookup

    # Create edge masks
    train_edge_mask = torch.tensor(
            [(u, v) in train_edges_set for u, v in combined_edges], dtype=torch.bool
        )
    test_edge_mask = ~train_edge_mask  # Complement mask

    # Create the unified data object
    unified_data = Data(
            x=data.x,
            edge_index=edge_index,
            y=data.y,
            train_mask=train_mask,
            test_mask=test_mask,
            train_edge_mask=train_edge_mask,
            test_edge_mask=test_edge_mask
        )

        # Logging
    print(f'Train graph: {train_mask.sum().item()} nodes, {len(train_edges)} edges')
    print(f'Test graph: {test_mask.sum().item()} nodes, {len(test_edges)} edges')
    # print(f'Train graph: {train_mask.sum().item()} nodes, {train_edge_mask.sum().item()} edges')
    # print(f'Test graph: {test_mask.sum().item()} nodes, {test_edge_mask.sum().item()} edges')
    return unified_data
        
 
   
def compute_class_distribution(train_labels, test_labels):
    import matplotlib.pyplot as plt

    # Ensure the tensors are on CPU and convert them to NumPy arrays
    train_labels_np = train_labels.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()
    
    # Calculate class frequencies for train dataset
    unique_train, counts_train = np.unique(train_labels_np, return_counts=True)
    train_distribution = dict(zip(unique_train, counts_train))
    
    # Calculate class frequencies for test dataset
    unique_test, counts_test = np.unique(test_labels_np, return_counts=True)
    test_distribution = dict(zip(unique_test, counts_test))
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    
    # Train distribution plot
    axs[0].bar(train_distribution.keys(), train_distribution.values())
    axs[0].set_title('Train Nodes Class Distribution')
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Frequency')
    
    # Test distribution plot
    axs[1].bar(test_distribution.keys(), test_distribution.values())
    axs[1].set_title('Test Nodes Class Distribution')
    axs[1].set_xlabel('Class')
    
    plt.show()



def generate_train_test_edge_only_data(original_data, train_indices, test_indices):
    """
    Generate a modified graph that contains only the edges between train and test nodes.

    Parameters:
    - original_data: PyTorch Geometric Data object containing the full graph.
    - train_indices: Tensor of node indices for the training nodes.
    - test_indices: Tensor of node indices for the test nodes.

    Returns:
    - modified_data: PyTorch Geometric Data object containing only the cross-train-test edges.
    """
    train_indices_set = set(train_indices.tolist())
    test_indices_set = set(test_indices.tolist())

    # Filter edges to include only those between train and test nodes
    train_test_edges = []
    for edge in original_data.edge_index.t().tolist():
        if (edge[0] in train_indices_set and edge[1] in test_indices_set) or \
           (edge[0] in test_indices_set and edge[1] in train_indices_set):
            train_test_edges.append(edge)
    
    # Convert the filtered edges to a tensor
    train_test_edge_index = torch.tensor(train_test_edges, dtype=torch.long).t().contiguous()

    # Create the modified data object with only cross-train-test edges
    modified_data = Data(x=original_data.x, edge_index=train_test_edge_index, y=original_data.y)

    return modified_data




def construct_random_node_with_neighbors(train_data, max_neighbors=5):
    """
    Construct a random node with:
    - Random feature values sampled from the feature space.
    - Random neighbors from the train set.
    - Assign the node a label corresponding to the class with the least probability predicted by the model.

    Parameters:
    - train_data: The training data containing features, edge indices, and labels.
    - model: The trained GNN model.

    Returns:
    - new_node_features: Tensor of the new node's features.
    - new_node_neighbors: Tensor of indices representing the new node's neighbors.
    - assigned_label: The label corresponding to the least probable class.
    """
    num_features = train_data.x.shape[1]
    num_nodes = train_data.x.shape[0]
    
    # Sample random feature values from the global distribution of features
    global_feature_distribution = train_data.x
    new_node_features = torch.empty(num_features)
    
    for i in range(num_features):
        new_node_features[i] = random.choice(global_feature_distribution[:, i].tolist())
    
    # Choose random neighbors from the train set
    num_neighbors = random.randint(1, max_neighbors)  # Let's say between 1 to 5 neighbors
    new_node_neighbors = random.sample(range(num_nodes), num_neighbors)
    
    # Use the model to predict the probabilities of each class for the new node
    new_node_features = new_node_features.unsqueeze(0)  # Make it 2D to pass through the model
    

    return new_node_features, new_node_neighbors

def add_corrupted_node_to_train_data(train_data, new_node_features, new_node_neighbors
                                    ):
    """
    Add a new corrupted node with random features and random neighbors to the training data.

    Parameters:
    - train_data: The training data containing features, edge indices, and labels.
    - new_node_features: Tensor of the new node's features.
    - new_node_neighbors: List of indices representing the new node's neighbors.
    - assigned_label: The label assigned to the new node.

    Returns:
    - corrupted_data: The corrupted training data containing the new node and its connections.
    """
    # Get the current number of nodes in the graph
    num_nodes = train_data.x.shape[0]
    
    # Extend the feature matrix by adding the new node's features
    new_features = torch.cat([train_data.x, new_node_features], dim=0)
    
    # Create new edges between the new node and its neighbors
    new_edges = [[num_nodes, neighbor] for neighbor in new_node_neighbors] + \
                [[neighbor, num_nodes] for neighbor in new_node_neighbors]  # Bidirectional edges
    new_edges = torch.tensor(new_edges, dtype=torch.long).t().contiguous()  # Format as edge index
    
    # Extend the edge index by adding the new edges
    new_edge_index = torch.cat([train_data.edge_index, new_edges], dim=1)

    # Assign a random label to the new node
    assigned_label = torch.randint(0, train_data.y.max().item() + 1, (1,)).item()
    
    # Extend the label tensor by adding the new node's label
    new_labels = torch.cat([train_data.y, torch.tensor([assigned_label], dtype=train_data.y.dtype)])
    
    # Create the corrupted dataset
    corrupted_data = Data(
        x=new_features,
        edge_index=new_edge_index,
        y=new_labels
    )
    
    return corrupted_data

def create_and_save_splits(data, num_splits=5, max_train_nodes=1500, max_neighbors=3, save_dir='splits'):
    """
    Create and save multiple random splits using snowball sampling.

    Parameters:
    - data: PyTorch Geometric Data object containing the graph.
    - num_splits: Number of random splits to generate and save (default is 5).
    - max_train_nodes: Maximum number of nodes in the training graph for each split.
    - max_neighbors: Maximum number of neighbors for snowball sampling.
    - save_path: Directory where the splits will be saved (default is 'splits/').
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    for split_num in range(1, num_splits + 1):
        # Step 1: Select initial nodes randomly from each class
        initial_nodes = select_random_nodes_from_each_class(data, 10)
        
        # Step 2: Perform snowball sampling to create train and test graphs
        train_data, test_data, train_to_orig, test_to_orig = snowball_sampling(data, initial_nodes, max_train_nodes, max_neighbors)

        # Print out the information for the current split
        print(f"Split {split_num}:")
        print(f"Number of nodes in train_data: {train_data.x.size(0)}, Number of edges: {train_data.edge_index.size(1)}")
        print(f"Number of nodes in test_data: {test_data.x.size(0)}, Number of edges: {test_data.edge_index.size(1)}")

        # Step 3: Save the split to a file using torch.save()
        split_filename = f'{save_dir}/split_{split_num}.pt'
        torch.save((train_data, test_data, train_to_orig, test_to_orig), split_filename)

def create_and_save_splits_joint(data, num_splits=5, max_neighbors=3, train_ratio=0.1, save_dir='splits', strategy='snowball'):

    print(f"save_path: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    for split_num in range(1, num_splits + 1):
        rng=random.Random(4000+split_num)
        if(strategy == 'snowball'):
            # Step 1: Select initial nodes randomly from each class
            initial_nodes = select_random_nodes_from_each_class(data, 10,rng)    
            # Step 2: Perform snowball sampling to create train and test graphs
            unified_data = snowball_sampling_with_train_test(data, initial_nodes, train_ratio, max_neighbors,rng)
        elif(strategy == 'random'):
            unified_data = create_inductive_random_splits(data, train_ratio)
        elif(strategy == 'snowball_induced'):
            # Step 1: Select initial nodes randomly from each class
            initial_nodes = select_random_nodes_from_each_class(data, 10,rng)    
            # Step 2: Perform snowball sampling to create train and test graphs
            unified_data = snowball_sampling_with_train_test_induced(data, initial_nodes, train_ratio, max_neighbors,rng)
     
        # Step 3: Save the split to a file using torch.save()
        split_filename = f'{save_dir}/split_{split_num}.pt'
        torch.save((unified_data), split_filename)
  
