from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.nn as nn

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='GCN'):
        super(GNNModel, self).__init__()
        
        # Select the appropriate GNN layer type
        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)

            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == 'GraphSage':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=True)
            self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        return nn.functional.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        # x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
def execute_function_by_name(module, func_name, *args, **kwargs):
      # Ensure func_name is a string
    if not isinstance(func_name, str):
        print(func_name)
        raise TypeError("Function name must be a string")
       

    # Retrieve the function object using getattr
    try:
        func = getattr(module, func_name)
        return func(*args, **kwargs)
    except AttributeError:
        return f"Function {func_name} not found in the module."

