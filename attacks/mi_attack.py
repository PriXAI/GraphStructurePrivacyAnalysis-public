import torch

from sklearn.metrics import accuracy_score, roc_auc_score,average_precision_score, precision_score, f1_score, recall_score, roc_curve
import attacks.attack_utils as atk
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import torch.nn as nn
import torch.optim as optim
import models.model as model

def train_membership_inference_attack_model(model, train_loader, num_epochs, learning_rate):
    """
    Trains the membership inference attack model using the provided training data.

    Parameters:
    - model: The membership inference attack model (usually a binary classification model).
    - train_loader: DataLoader containing the training data (features and labels).
    - num_epochs: Number of epochs for training.
    - learning_rate: Learning rate for the optimizer.

    Returns:
    - Trained model.
    """
    # Binary cross-entropy loss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Set the model in training mode
        model.train()

        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass: compute model predictions
            outputs = model(inputs)
            predicted_probs = torch.sigmoid(outputs)
            
            # Reshape labels to match the shape of outputs
            labels = labels.view(-1, 1).float()  # Ensure labels are float for BCELoss
            
            # Compute the loss
            loss = criterion(predicted_probs, labels)
            
            # Backward pass: compute the gradient of the loss w.r.t. the model parameters
            loss.backward()
            
            # Perform a single optimization step (update model parameters)
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()

        # Print the average loss for the current epoch
        avg_loss = running_loss / len(train_loader)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("Training complete!")
    return model

def evaluate_membership_inference_attack_model(model, test_loader):
    """
    Evaluate a membership inference attack model using accuracy and AUROC metrics.
    
    Parameters:
    - model: The trained membership inference attack model.
    - test_loader: DataLoader containing the test dataset (inputs and labels).

    Returns:
    - accuracy: Accuracy of the model.
    - auroc: AUROC (Area Under the Receiver Operating Characteristic) score of the model.
    """
    model.eval()  # Set model to evaluation mode
    y_true = []  # List to store true labels
    y_pred = []  # List to store predicted binary labels
    predicted_probs_all = []  # List to store predicted probabilities for AUROC computation
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader: 
            outputs = model(inputs)  # Forward pass to get raw logits
            
            # Apply sigmoid to get probabilities
            predicted_probs = torch.sigmoid(outputs)
            
            
            # Convert probabilities to binary labels (>= 0.5 considered as class 1)
            predicted_labels = (predicted_probs >= 0.5).int()  
            
            # Convert the labels and predictions to lists of integers
            y_true.extend(labels.cpu().numpy().astype(int).tolist())  # Convert true labels to integers
            y_pred.extend(predicted_labels.cpu().numpy().astype(int).tolist())  # Convert predicted labels to integers
            
            # Collect predicted probabilities for AUROC
            predicted_probs_all.extend(predicted_probs.cpu().numpy().tolist())
    
    # print("true labels:",y_true)
    
    y_pred_flat = [pred[0] for pred in y_pred]
    predicted_probs_flat = [pred[0] for pred in predicted_probs_all]
    # print("Predicted probabilities:",predicted_probs_flat)
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred_flat)
    # print("Prediction list:",y_pred)
    # count the number of true positives
    predicted_positives = sum(1 for pred in y_pred_flat if pred == 1)
    # true_positives = sum(1 for (pred_true,pred) in (y_true,y_pred) if pred[0] == [1] and pred_true[0] == [1])
    print("Number of predicted positives:",predicted_positives)
    # print("Number of true positives:",true_positives)
    precision=precision_score(y_true, y_pred_flat)
    f1=f1_score(y_true, y_pred_flat)
  

    # Compute AUROC (if y_true contains both 0s and 1s, otherwise AUROC is undefined)
    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, predicted_probs_flat)
        # Compute FPR, TPR, and thresholds
        fpr, tpr, thresholds = roc_curve(y_true, predicted_probs_flat)
        # Membership Advantage: max difference between TPR and FPR
        membership_advantage = np.max(tpr - fpr)
    else:
        auroc = 'AUROC is undefined (only one class present in test labels)'

    #Compute average precision
    ap=average_precision_score(y_true, predicted_probs_flat)
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC Score: {auroc}")
    print(f"Average Precision Score: {ap}")
    print(f"Precision Score: {precision}")
    print(f"F1 Score: {f1}")
    print("Recall:", recall_score(y_true, y_pred))
    
    return accuracy, auroc, ap, precision, f1, membership_advantage

def run_attack_for_splits(
    splits,
    posteriors,
    labels,
    train_indices,
    test_indices,
    num_epochs=200,
    learning_rate=0.001,
):
    """
    Run the attack model training and evaluation on each split.
    """
    accuracies, aucs, aps, precisions, f1s, mas = [], [], [], [], [], []
    
    
    features, attack_labels = atk.construct_attack_features(posteriors, labels, train_indices, test_indices)
    
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Processing split {i + 1}/{len(splits)}")

        # Split features and labels based on the current split
        train_features, test_features = features[train_idx], features[test_idx]
        train_labels, test_labels = attack_labels[train_idx], attack_labels[test_idx]
        print("Number of train nodes with positive class in the train set:", int(train_labels.sum().item()))
        #count the numebr of test nodes with postive class in test data
        print("Number of test nodes with positive class in the test set:", int(test_labels.sum().item()))
    

        # Create DataLoader for training and testing
        train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=32, shuffle=False)

        # Initialize the attack model (simple MLP)
        input_dim = train_features.shape[1]
        hidden_dim = 64
        output_dim = 1
        
        attack_model = model.MLP(input_dim, hidden_dim, output_dim)

        # Train and evaluate the attack model
        attack_model = train_membership_inference_attack_model(attack_model, train_loader, num_epochs, learning_rate)
        accuracy, auc, ap, precision, f1, membership_advantage = evaluate_membership_inference_attack_model(attack_model, test_loader)

        # Store results
        accuracies.append(accuracy)
        aucs.append(auc)
        aps.append(ap)
        precisions.append(precision)
        f1s.append(f1)
        mas.append(membership_advantage)


    return accuracies, aucs, aps, precisions, f1s, mas
