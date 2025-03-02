
import torch
import torch.nn as nn
from mtgnn import MTGNN

def create_safe_mtgnn_model(num_nodes, in_dim, seq_length=4, device='cuda'):
    """
    Create a MTGNN model with safe configuration for short sequences
    """
    # Use conservative settings for short sequences
    model_config = {
        'gcn_true': True,
        'build_adj': True,
        'gcn_depth': 1,               # Reduced depth
        'num_nodes': num_nodes,
        'kernel_set': [2],            # Minimal kernel size
        'kernel_size': 2,             # Minimal kernel size
        'dropout': 0.3,
        'subgraph_size': 3,
        'node_dim': 20,
        'dilation_exponential': 1,    # No dilation
        'conv_channels': 16,
        'residual_channels': 16,
        'skip_channels': 32,
        'end_channels': 64,
        'seq_length': seq_length,
        'in_dim': in_dim,
        'out_dim': 2,
        'layers': 1,                  # Single layer
        'propalpha': 0.05,
        'tanhalpha': 3,
        'layer_norm_affline': True
    }
    
    print(f"Creating safe MTGNN model with sequence length {seq_length}")
    
    # Initialize model
    model = MTGNN(**model_config).to(device)
    
    # Safety check for any remaining problematic kernels
    fix_model_kernels(model, seq_length)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model

def fix_model_kernels(model, seq_length):
    """
    Fix any remaining problematic kernels in the model
    """
    max_kernel_size = min(seq_length, 3)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.kernel_size[1] > max_kernel_size:
                print(f"Fixing kernel in {name}: {module.kernel_size} -> (1, {max_kernel_size})")
                
                # Create new convolution with smaller kernel
                new_conv = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=(1, max_kernel_size),
                    bias=module.bias is not None
                ).to(module.weight.device)
                
                # Initialize weights
                nn.init.xavier_uniform_(new_conv.weight)
                if new_conv.bias is not None and module.bias is not None:
                    new_conv.bias.data = module.bias.data[:module.out_channels]
                
                # Replace module
                if '.' in name:
                    parent_name, child_name = name.rsplit('.', 1)
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, new_conv)
                else:
                    setattr(model, name, new_conv)

