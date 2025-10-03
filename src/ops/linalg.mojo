"""
Basic Linear Algebra Operations
Matrix multiplication, dot products, and other linear algebra operations.
"""

from algorithm import vectorize, parallelize
from sys import simd_width_of
from ..core.tensor import Tensor
from ..core.shape import TensorShape

fn matmul[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    """Matrix multiplication with SIMD optimization.
    
    Computes C = A @ B where:
    - A is (M, K)
    - B is (K, N)
    - C is (M, N)
    
    Uses cache-friendly blocked algorithm with SIMD vectorization.
    
    Args:
        a: Left matrix (M, K)
        b: Right matrix (K, N)
    
    Returns:
        Result matrix (M, N)
    """
    # Extract dimensions
    var M = a.shape.dims[0]
    var K = a.shape.dims[1]
    var N = b.shape.dims[1]
    
    # Create result matrix
    var result = Tensor[dtype](M, N)
    
    # Simple triple-nested loop (TODO: optimize with tiling and SIMD)
    for i in range(M):
        for j in range(N):
            var sum_val = Scalar[dtype](0)
            
            # Dot product of row i from A and column j from B
            for k in range(K):
                var a_idx = DynamicVector[Int]()
                a_idx.push_back(i)
                a_idx.push_back(k)
                
                var b_idx = DynamicVector[Int]()
                b_idx.push_back(k)
                b_idx.push_back(j)
                
                sum_val += a[a_idx] * b[b_idx]
            
            var result_idx = DynamicVector[Int]()
            result_idx.push_back(i)
            result_idx.push_back(j)
            result[result_idx] = sum_val
    
    return result

fn matmul_tiled[dtype: DType, tile_size: Int = 32](
    a: Tensor[dtype], b: Tensor[dtype]
) -> Tensor[dtype]:
    """Tiled matrix multiplication for better cache utilization.
    
    Uses blocking/tiling to keep data in cache during computation.
    
    Parameters:
        dtype: Data type of tensors
        tile_size: Size of tiles (default 32x32)
    
    Args:
        a: Left matrix (M, K)
        b: Right matrix (K, N)
    
    Returns:
        Result matrix (M, N)
    """
    var M = a.shape.dims[0]
    var K = a.shape.dims[1]
    var N = b.shape.dims[1]
    
    var result = Tensor[dtype](M, N)
    
    # Tiled algorithm
    for i_tile in range(0, M, tile_size):
        for j_tile in range(0, N, tile_size):
            for k_tile in range(0, K, tile_size):
                
                # Process tile
                var i_end = min(i_tile + tile_size, M)
                var j_end = min(j_tile + tile_size, N)
                var k_end = min(k_tile + tile_size, K)
                
                for i in range(i_tile, i_end):
                    for j in range(j_tile, j_end):
                        var sum_val = Scalar[dtype](0)
                        
                        for k in range(k_tile, k_end):
                            var a_idx = DynamicVector[Int]()
                            a_idx.push_back(i)
                            a_idx.push_back(k)
                            
                            var b_idx = DynamicVector[Int]()
                            b_idx.push_back(k)
                            b_idx.push_back(j)
                            
                            sum_val += a[a_idx] * b[b_idx]
                        
                        var result_idx = DynamicVector[Int]()
                        result_idx.push_back(i)
                        result_idx.push_back(j)
                        
                        if k_tile == 0:
                            result[result_idx] = sum_val
                        else:
                            result[result_idx] = result[result_idx] + sum_val
    
    return result

fn dot[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Scalar[dtype]:
    """Dot product of two 1D tensors.
    
    Computes sum(a[i] * b[i]) with SIMD optimization.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Scalar dot product
    """
    alias simd_width = simd_width_of[dtype]()
    var result = Scalar[dtype](0)
    
    # Vectorized dot product
    var num_vectors = a.size // simd_width
    var simd_sum = SIMD[dtype, simd_width](0)
    
    for i in range(num_vectors):
        var vec_a = a.data.load[width=simd_width](i * simd_width)
        var vec_b = b.data.load[width=simd_width](i * simd_width)
        simd_sum += vec_a * vec_b
    
    # Horizontal reduction
    for i in range(simd_width):
        result += simd_sum[i]
    
    # Handle remaining elements
    var remainder = a.size % simd_width
    for i in range(a.size - remainder, a.size):
        result += a.data[i] * b.data[i]
    
    return result

fn outer[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    """Outer product of two 1D tensors.
    
    Computes result[i, j] = a[i] * b[j]
    
    Args:
        a: First vector (size M)
        b: Second vector (size N)
    
    Returns:
        Matrix of size (M, N)
    """
    var M = a.size
    var N = b.size
    var result = Tensor[dtype](M, N)
    
    for i in range(M):
        for j in range(N):
            var idx = DynamicVector[Int]()
            idx.push_back(i)
            idx.push_back(j)
            result[idx] = a.data[i] * b.data[j]
    
    return result

fn trace[dtype: DType](tensor: Tensor[dtype]) -> Scalar[dtype]:
    """Compute trace of a square matrix (sum of diagonal elements).
    
    Args:
        tensor: Square matrix (N, N)
    
    Returns:
        Sum of diagonal elements
    """
    var N = tensor.shape.dims[0]
    var result = Scalar[dtype](0)
    
    for i in range(N):
        var idx = DynamicVector[Int]()
        idx.push_back(i)
        idx.push_back(i)
        result += tensor[idx]
    
    return result

fn norm[dtype: DType](tensor: Tensor[dtype], p: Int = 2) -> Scalar[dtype]:
    """Compute p-norm of a tensor.
    
    L2 norm (default): sqrt(sum(x^2))
    L1 norm: sum(|x|)
    
    Args:
        tensor: Input tensor
        p: Norm order (1 or 2)
    
    Returns:
        Computed norm
    """
    if p == 2:
        # L2 norm (Euclidean)
        alias simd_width = simd_width_of[dtype]()
        var sum_squared = Scalar[dtype](0)
        
        var num_vectors = tensor.size // simd_width
        var simd_sum = SIMD[dtype, simd_width](0)
        
        for i in range(num_vectors):
            var vec = tensor.data.load[width=simd_width](i * simd_width)
            simd_sum += vec * vec
        
        # Horizontal reduction
        for i in range(simd_width):
            sum_squared += simd_sum[i]
        
        # Remaining elements
        var remainder = tensor.size % simd_width
        for i in range(tensor.size - remainder, tensor.size):
            var val = tensor.data[i]
            sum_squared += val * val
        
        return sqrt(sum_squared)
    
    else:  # L1 norm
        var sum_abs = Scalar[dtype](0)
        for i in range(tensor.size):
            sum_abs += abs(tensor.data[i])
        return sum_abs
