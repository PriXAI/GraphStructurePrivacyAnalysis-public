import torch.nn as nn
import torch

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute loss
    loss = nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    train_acc,test_acc= evaluate_acc(model, data)
    #print(f'Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    
    return loss.item()

def evaluate(model, train_data):
    model.eval()
    with torch.no_grad():
        out = model(train_data.x, train_data.edge_index)
        loss = nn.functional.nll_loss(out, train_data.y)
    return loss.item()

def evaluate_acc(model,data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        train_acc= (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean()
        test_acc= (out[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).float().mean()
    return train_acc.item(), test_acc.item()

def train_with_extra_iterations(model, optimizer, train_data, target_class, epochs, extra_iterations):
    """
    Train the model with additional iterations over nodes of a particular class.

    Parameters:
    - model: The neural network model.
    - optimizer: The optimizer.
    - train_data: The dataset, including features and labels.
    - target_class: The class over which to perform extra iterations.
    - epochs: Number of epochs to train.
    - extra_iterations: Number of extra iterations for the target class.
    """
    model.train()
    target_indices = (train_data.y == target_class).nonzero().squeeze()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)  # Example forward pass, adjust according to your model
        loss = torch.nn.functional.cross_entropy(out, train_data.y)
        loss.backward()
        optimizer.step()

        # Extra iterations over the target class
        for i in range(extra_iterations):
            optimizer.zero_grad()
            out = model(train_data.x, train_data.edge_index)  # Forward pass
            target_loss = torch.nn.functional.cross_entropy(out[target_indices], train_data.y[target_indices])
            target_loss.backward()
            optimizer.step()

            print(f'iteration {i+1}, Loss: {loss.item()}, Target Class Loss: {target_loss.item()}')