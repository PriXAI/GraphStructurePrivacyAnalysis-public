import torch
import torch.nn.functional as F
import numpy as np

def get_posterior(model, data):
    """
    Helper function to call the various types of models in the right way for inference
    """
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
    # Apply softmax to get probabilities
    output = F.softmax(output, dim=1)  # or use log_softmax if you need log-probabilities
    return output


def compute_loss_without_edges(model, data):
    """
    Compute the loss using only node features and a null edge index.

    Parameters:
    - model: The trained GNN model.
    - data: PyTorch Geometric Data object containing the graph.

    Returns:
    - loss: The computed loss using only node features.
    """
    
    # Create a null edge index
    null_edge_index = torch.empty(2, 0, dtype=torch.long)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        # Forward pass using the node features and null edge index
        output = model(data.x, null_edge_index)
    
    # Compute the loss
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    losses = criterion(output, data.y)

    return losses

# Compute posteriors
def compute_posterior_without_edges(model, data):
    """
    Computes posterior probabilities for all nodes in the graph using the trained model.
    
    Parameters:
    - model: The trained GNN model.
    - data: The PyTorch Geometric Data object containing the graph.
    
    Returns:
    - probs: Tensor containing the posterior probabilities for each node.
    """
    # Create a null edge index
    null_edge_index = torch.empty(2, 0, dtype=torch.long)
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        out = model(data.x, null_edge_index)  # Forward pass to get logits
    
    probs = F.softmax(out, dim=1)  # Apply log softmax to get log-probabilities
    # or use softmax if you need probabilities instead of log-probabilities
    # probs = F.softmax(out, dim=1)

    return probs

def compute_entropy(posteriors):
    """
    Compute the entropy of posteriors for each node.

    Parameters:
    - posteriors: A tensor or numpy array of shape (num_nodes, num_classes) representing
                  the posterior probabilities for each class of each node.

    Returns:
    - entropies: A tensor of entropies for each node.
    """
   
    
    # Compute entropy for each node
    entropies = -torch.sum(posteriors * np.log(posteriors + 1e-9), axis=1)  # Adding a small value to avoid log(0)

    return entropies

# TODO refactoring - the computation of accuracy can be unified across compute_accuracy, compute_accuracy_alledges and noedges
# move the print statements to the main code

def compute_accuracy(posterior, unified_data):
    
    pred = posterior.max(dim=1)[1]
    
    train_correct = pred[unified_data.train_mask].eq(unified_data.y[unified_data.train_mask]).sum().item()
    
    total = unified_data.train_mask.sum().item()
    train_acc = train_correct / total if total > 0 else 0.0
   
    test_correct = pred[unified_data.test_mask].eq(unified_data.y[unified_data.test_mask]).sum().item()
    total = unified_data.test_mask.sum().item()
    test_acc = test_correct / total if total > 0 else 0.0

    return train_acc,test_acc


