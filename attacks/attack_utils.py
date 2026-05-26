import torch
import torch.nn.functional as F
import random
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,accuracy_score,precision_score, recall_score
import warnings
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split



# Compute posteriors
def compute_posterior(model, data):
    """
    Computes posterior probabilities for all nodes in the graph using the trained model.
    
    Parameters:
    - model: The trained GNN model.
    - data: The PyTorch Geometric Data object containing the graph.
    
    Returns:
    - probs: Tensor containing the posterior probabilities for each node.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        out = model(data.x, data.edge_index)  # Forward pass to get logits
        
        probs = F.softmax(out, dim=1)  # Apply log softmax to get log-probabilities
        # or use softmax if you need probabilities instead of log-probabilities
        # probs = F.softmax(out, dim=1)
        
    return probs

def compute_posterior_inductive(model, train_data, test_data, train_to_orig, test_to_orig):
    """
    Compute posteriors for train and test data and map them back to original node indices.

    Parameters:
    - model: The trained GNN model.
    - train_data: PyTorch Geometric Data object for the training subgraph.
    - test_data: PyTorch Geometric Data object for the test subgraph.
    - train_to_orig: Dictionary mapping train graph node indices to original node indices.
    - test_to_orig: Dictionary mapping test graph node indices to original node indices.
  

    Returns:
    - orig_posteriors: Tensor of shape [total_nodes, num_classes] with the posteriors for each original node.
    """
    
    # Compute posteriors for train and test data
    posteriors_train = compute_posterior(model, train_data)
    posteriors_test = compute_posterior(model, test_data)
    
    # Determine the number of classes
    num_classes = posteriors_train.size(1)
    
    total_nodes= max(max(train_to_orig.values()), max(test_to_orig.values()))+1

    print("Total nodes:",total_nodes)

    # Initialize a tensor to hold the posteriors for all original nodes
    orig_posteriors = torch.zeros((total_nodes, num_classes))
    
    # Map train posteriors to the original indices
    for i, posterior in enumerate(posteriors_train):
        orig_index = train_to_orig[i]
        orig_posteriors[orig_index] = posterior 

    # Map test posteriors to the original indices
    for j, posterior in enumerate(posteriors_test):
        orig_index = test_to_orig[j]
        orig_posteriors[orig_index] = posterior 

    return orig_posteriors


def save_splits(splits, filename="splits.pkl"):
    """
    Save the splits to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(splits, f)

def load_splits(filename="splits.pkl"):
    """
    Load saved splits from a file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def construct_attack_labels(num_nodes, train_indices, test_indices, device=None):
    """
    Build binary membership labels for the attack dataset.

    Members (target-model train nodes) are labeled 1 and
    non-members (target-model test nodes) are labeled 0.
    """
    attack_labels = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    attack_labels[train_indices] = 1.0
    attack_labels[test_indices] = 0.0
    return attack_labels

def create_random_splits(node_indices, num_splits=5, test_size=0.3, seed=42, stratify_labels=None):
    """
    Create multiple random splits for train and test sets using a fixed seed for reproducibility.
    
    Parameters:
    - node_indices: Tensor of node indices.
    - num_splits: Number of random splits to create.
    - test_size: Fraction of the data to use for the test set.
    - seed: Random seed for reproducibility.
    - stratify_labels: Optional labels used to preserve class balance across
      the generated train/test splits.
    
    Returns:
    - splits: A list of (train_indices, test_indices) tuples.
    """
    splits = []
    
    # Set random seed to ensure reproducibility
    random.seed(seed)
    node_indices_np = node_indices.cpu().numpy() if isinstance(node_indices, torch.Tensor) else np.asarray(node_indices)
    stratify_np = None
    if stratify_labels is not None:
        stratify_np = stratify_labels.cpu().numpy() if isinstance(stratify_labels, torch.Tensor) else np.asarray(stratify_labels)
    
    for i in range(num_splits):
        # Pass a different random_state for each split using the seed and split index
        train_idx, test_idx = train_test_split(
            node_indices_np,
            test_size=test_size,
            random_state=seed + i,
            stratify=stratify_np,
        )
        if isinstance(node_indices, torch.Tensor):
            train_idx = torch.as_tensor(train_idx, dtype=node_indices.dtype, device=node_indices.device)
            test_idx = torch.as_tensor(test_idx, dtype=node_indices.dtype, device=node_indices.device)
        splits.append((train_idx, test_idx))
    
    return splits


def construct_attack_features(posteriors, labels, train_indices, test_indices):
    """
    Construct attack features by concatenating posteriors with per-node loss values.
    Labels are 1 for training nodes and 0 for test nodes.

    Parameters:
    - posteriors: Posterior probabilities for all nodes.
    - labels: True labels for all nodes.
    - train_indices: Indices of training nodes.
    - test_indices: Indices of test nodes.

    Returns:
    - attack_features: Concatenated posteriors and losses for all nodes.
    - attack_labels: 1 for training nodes, 0 for test nodes.
    """
    node_indices = torch.arange(posteriors.shape[0], device=posteriors.device)
    true_labels = labels.to(posteriors.device).long()
    true_class_probs = posteriors[node_indices, true_labels].clamp_min(1e-12)
    losses = -torch.log(true_class_probs)
    attack_features = torch.cat((posteriors, losses.unsqueeze(1)), dim=1)

    
    # Create labels: 1 for training nodes, 0 for test nodes
    attack_labels = construct_attack_labels(
        posteriors.shape[0],
        train_indices,
        test_indices,
        device=posteriors.device,
    )

    return attack_features, attack_labels

def calculate_average_precision_auroc_with_scores(predicted_scores, true_scores):
    # Error handling: Check if all labels are the same (all 0s or all 1s)
    if all(label == 0 for label in true_scores) or all(label == 1 for label in true_scores):
        warnings.warn("All labels are the same. ROC AUC and Average Precision scores may not be meaningful.")
    
    # Compute average precision and ROC AUC using sklearn's functions
    try:
        ap = average_precision_score(true_scores, predicted_scores)
        print("AP Score:",ap)
    except ValueError as e:
        raise ValueError(f"Error calculating average precision: {e}")
    
    try:
        auc_score = roc_auc_score(true_scores, predicted_scores)
        print("AUC Score:",auc_score)
    except ValueError as e:
        raise ValueError(f"Error calculating ROC AUC score: {e}")
    
    return ap, auc_score
