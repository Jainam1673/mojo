"""
Reduction Operations for Tensors
Operations that reduce tensor dimensions (sum, mean, max, min, etc.)
"""

from algorithm import vectorize
from sys import simdwidthof
from ..core.tensor import Tensor
from math import max as math_max, min as math_min

fn sum[dtype: DType](tensor: Tensor[dtype]) -> Scalar[dtype]:
    """Sum all elements in the tensor.
    
    Uses SIMD vectorization and horizontal reduction for maximum performance.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Sum of all elements
    """
    alias simd_width = simdwidthof[dtype]()
    var total = Scalar[dtype](0)
    
    # Vectorized accumulation
    var num_vectors = tensor.size // simd_width
    var simd_sum = SIMD[dtype, simd_width](0)
    
    for i in range(num_vectors):
        var vec = tensor.data.load[width=simd_width](i * simd_width)
        simd_sum += vec
    
    # Horizontal reduction of SIMD vector
    for i in range(simd_width):
        total += simd_sum[i]
    
    # Handle remaining elements
    var remainder = tensor.size % simd_width
    for i in range(tensor.size - remainder, tensor.size):
        total += tensor.data[i]
    
    return total

fn mean[dtype: DType](tensor: Tensor[dtype]) -> Scalar[dtype]:
    """Calculate mean (average) of all elements.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Mean of all elements
    """
    var total = sum(tensor)
    return total / Scalar[dtype](tensor.size)

fn max[dtype: DType](tensor: Tensor[dtype]) -> Scalar[dtype]:
    """Find maximum element in the tensor.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Maximum element
    """
    alias simd_width = simdwidthof[dtype]()
    var num_vectors = tensor.size // simd_width
    
    # Initialize with first value
    var max_val = tensor.data[0]
    
    if num_vectors > 0:
        var simd_max = tensor.data.load[width=simd_width](0)
        
        # Vectorized max finding
        for i in range(1, num_vectors):
            var vec = tensor.data.load[width=simd_width](i * simd_width)
            simd_max = max(simd_max, vec)
        
        # Horizontal reduction
        for i in range(simd_width):
            max_val = math_max(max_val, simd_max[i])
    
    # Handle remaining elements
    var remainder = tensor.size % simd_width
    for i in range(tensor.size - remainder, tensor.size):
        max_val = math_max(max_val, tensor.data[i])
    
    return max_val

fn min[dtype: DType](tensor: Tensor[dtype]) -> Scalar[dtype]:
    """Find minimum element in the tensor.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Minimum element
    """
    alias simd_width = simdwidthof[dtype]()
    var num_vectors = tensor.size // simd_width
    
    # Initialize with first value
    var min_val = tensor.data[0]
    
    if num_vectors > 0:
        var simd_min = tensor.data.load[width=simd_width](0)
        
        # Vectorized min finding
        for i in range(1, num_vectors):
            var vec = tensor.data.load[width=simd_width](i * simd_width)
            simd_min = min(simd_min, vec)
        
        # Horizontal reduction
        for i in range(simd_width):
            min_val = math_min(min_val, simd_min[i])
    
    # Handle remaining elements
    var remainder = tensor.size % simd_width
    for i in range(tensor.size - remainder, tensor.size):
        min_val = math_min(min_val, tensor.data[i])
    
    return min_val

fn variance[dtype: DType](tensor: Tensor[dtype]) -> Scalar[dtype]:
    """Calculate variance of all elements.
    
    Var(X) = E[(X - μ)²] where μ is the mean
    
    Args:
        tensor: Input tensor
    
    Returns:
        Variance of elements
    """
    var mu = mean(tensor)
    var sum_squared_diff = Scalar[dtype](0)
    
    alias simd_width = simdwidthof[dtype]()
    
    @parameter
    fn variance_vectorized[width: Int](i: Int):
        var vec = tensor.data.load[width=width](i)
        var diff = vec - mu
        var squared = diff * diff
        
        # Horizontal sum
        for j in range(width):
            sum_squared_diff += squared[j]
    
    vectorize[variance_vectorized, simd_width](tensor.size)
    
    return sum_squared_diff / Scalar[dtype](tensor.size)

fn std[dtype: DType](tensor: Tensor[dtype]) -> Scalar[dtype]:
    """Calculate standard deviation of all elements.
    
    Std(X) = √Var(X)
    
    Args:
        tensor: Input tensor
    
    Returns:
        Standard deviation of elements
    """
    return sqrt(variance(tensor))

fn prod[dtype: DType](tensor: Tensor[dtype]) -> Scalar[dtype]:
    """Product of all elements in the tensor.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Product of all elements
    """
    alias simd_width = simdwidthof[dtype]()
    var total = Scalar[dtype](1)
    
    # Vectorized multiplication
    var num_vectors = tensor.size // simd_width
    var simd_prod = SIMD[dtype, simd_width](1)
    
    for i in range(num_vectors):
        var vec = tensor.data.load[width=simd_width](i * simd_width)
        simd_prod *= vec
    
    # Horizontal reduction
    for i in range(simd_width):
        total *= simd_prod[i]
    
    # Handle remaining elements
    var remainder = tensor.size % simd_width
    for i in range(tensor.size - remainder, tensor.size):
        total *= tensor.data[i]
    
    return total

fn argmax[dtype: DType](tensor: Tensor[dtype]) -> Int:
    """Find index of maximum element.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Linear index of maximum element
    """
    var max_val = tensor.data[0]
    var max_idx = 0
    
    for i in range(1, tensor.size):
        if tensor.data[i] > max_val:
            max_val = tensor.data[i]
            max_idx = i
    
    return max_idx

fn argmin[dtype: DType](tensor: Tensor[dtype]) -> Int:
    """Find index of minimum element.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Linear index of minimum element
    """
    var min_val = tensor.data[0]
    var min_idx = 0
    
    for i in range(1, tensor.size):
        if tensor.data[i] < min_val:
            min_val = tensor.data[i]
            min_idx = i
    
    return min_idx
