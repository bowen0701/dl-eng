import torch
import torch.nn as nn
import numpy as np
from dl_eng.models.manual_layers import Linear, ReLU

def test_linear_layer_backward():
    """Verify Linear layer gradients against PyTorch autograd."""
    batch_size, in_features, out_features = 2, 3, 2
    
    # Input data
    x_np = np.random.randn(batch_size, in_features).astype(np.float32)
    x_torch = torch.from_numpy(x_np).requires_grad_(True)
    
    # Setup manual and torch layers
    linear_manual = Linear(in_features, out_features)
    linear_torch = nn.Linear(in_features, out_features)
    
    # Sync weights
    with torch.no_grad():
        linear_torch.weight.copy_(torch.from_numpy(linear_manual.W))
        linear_torch.bias.copy_(torch.from_numpy(linear_manual.b.flatten()))
        
    # Forward
    z_manual, cache_linear = linear_manual.forward(x_np)
    z_torch = linear_torch(x_torch)
    
    assert np.allclose(z_manual, z_torch.detach().numpy(), atol=1e-5)
    
    # Backward
    dz_np = np.random.randn(*z_manual.shape).astype(np.float32)
    dz_torch = torch.from_numpy(dz_np)
    
    dx_manual = linear_manual.backward(dz_np, cache_linear)
    z_torch.backward(dz_torch)
    
    # Assertions
    assert np.allclose(linear_manual.dW, linear_torch.weight.grad.numpy(), atol=1e-5)
    assert np.allclose(linear_manual.db.flatten(), linear_torch.bias.grad.numpy(), atol=1e-5)
    assert np.allclose(dx_manual, x_torch.grad.numpy(), atol=1e-5)

def test_relu_layer_backward():
    """Verify ReLU layer gradients against PyTorch autograd."""
    batch_size, features = 4, 5
    
    # Input data
    z_np = np.random.randn(batch_size, features).astype(np.float32)
    z_torch = torch.from_numpy(z_np).requires_grad_(True)
    
    # Setup
    relu_manual = ReLU()
    relu_torch = nn.ReLU()
    
    # Forward
    a_manual, cache_relu = relu_manual.forward(z_np)
    a_torch = relu_torch(z_torch)
    
    assert np.allclose(a_manual, a_torch.detach().numpy(), atol=1e-5)
    
    # Backward
    da_np = np.random.randn(*a_manual.shape).astype(np.float32)
    a_torch.backward(torch.from_numpy(da_np))
    
    dz_manual = relu_manual.backward(da_np, cache_relu)
    
    assert np.allclose(dz_manual, z_torch.grad.numpy(), atol=1e-5)
