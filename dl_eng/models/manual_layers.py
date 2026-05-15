import numpy as np
from typing import Tuple, Dict, Any

class Linear:
    """
    Linear Layer with manual backward pass implementation.
	
	Architecture Note: Stateless/Functional Pattern
	----------------------------------------------
	This implementation follows a stateless pattern where the layer does not 
	internally store the forward pass data (X). Instead, it returns a 'cache'.
	
	Why this pattern?
	1. Thread-Safety: Allows the same layer instance to be used in parallel.
	2. Weight Sharing: The same layer can be called multiple times in a single 
	   computational graph without overwriting internal state.
	3. Explicit Backprop: Clearly demonstrates exactly which tensors from the 
	   forward pass are required to compute gradients.
	
	Who 'catches' the cache? 
	In a full framework, a 'Model' or 'Sequential' container stores these 
	caches in a list during forward() and provides them back to the layers 
	in reverse order during backward().
	"""
    
    def __init__(self, in_features: int, out_features: int):
        # He initialization for ReLU networks
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        
        # Gradients stored after backward()
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward: Z = X W^T + b

        Shapes: X: (N, D_in), W: (D_out, D_in), b: (1, D_out), Z: (N, D_out)
        """
        Z = np.dot(X, self.W.T) + self.b
        cache = {"X": X}
        return Z, cache

    def backward(self, dZ: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Backprop (chain rule through Z = X W^T + b):
            dL/dW = dZ^T @ X        shape: (D_out, D_in)
            dL/db = sum(dZ, axis=0) shape: (1, D_out)
            dL/dX = dZ @ W          shape: (N, D_in)
        """
        X = cache["X"]
        batch_size = X.shape[0]

        # 1. Gradient wrt Weights: (out, batch) @ (batch, in) -> (out, in)
        self.dW = np.dot(dZ.T, X)
        
        # 2. Gradient wrt Bias: sum across batch
        self.db = np.sum(dZ, axis=0, keepdims=True)
        
        # 3. Gradient wrt Input (to pass to previous layer): (batch, out) @ (out, in) -> (batch, in)
        dX = np.dot(dZ, self.W)
        
        return dX

class ReLU:
    """
    ReLU Activation with manual backward pass implementation.

    Element-wise operation: input and output shapes are identical.
    """
    def forward(self, Z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward: A = max(0, Z)

        Shapes: Z: (N, D), A: (N, D)
        """
        A = np.maximum(0, Z)
        cache = {"Z": Z}
        return A, cache

    def backward(self, dA: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Backprop: dL/dZ = dL/dA * 1(Z > 0)   (gate: pass gradient where Z > 0, else 0)

        Shapes: dA: (N, D), dZ: (N, D)
        """
        Z = cache["Z"]
        dZ = dA.copy()
        # Gradient is 0 where Z was <= 0, otherwise it's 1 * dA
        dZ[Z <= 0] = 0
        return dZ
