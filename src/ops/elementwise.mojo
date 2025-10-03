"""
Element-wise Operations for Tensors
SIMD-optimized operations that work element-by-element.
"""

from algorithm import vectorize
from sys import simd_width_of
from ..core.tensor import Tensor
from ..core.shape import broadcast_shapes

# =============================================================================
# Element-wise Binary Operations
# =============================================================================

fn add[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise addition with SIMD vectorization.
    
    Supports broadcasting if shapes are compatible.
    
    Args:
        a: First tensor
        b: Second tensor
    
    Returns:
        Result tensor (a + b)
    """
    # TODO: Implement broadcasting
    # For now, assume same shape
    
    var result = Tensor[dtype](a.shape)
    alias simd_width = simd_width_of[dtype]()
    
    @parameter
    fn add_vectorized[width: Int](i: Int):
        var vec_a = a.data.load[width=width](i)
        var vec_b = b.data.load[width=width](i)
        result.data.store[width=width](i, vec_a + vec_b)
    
    vectorize[add_vectorized, simd_width](result.size)
    
    return result

fn sub[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise subtraction with SIMD vectorization.
    
    Args:
        a: First tensor
        b: Second tensor
    
    Returns:
        Result tensor (a - b)
    """
    var result = Tensor[dtype](a.shape)
    alias simd_width = simd_width_of[dtype]()
    
    @parameter
    fn sub_vectorized[width: Int](i: Int):
        var vec_a = a.data.load[width=width](i)
        var vec_b = b.data.load[width=width](i)
        result.data.store[width=width](i, vec_a - vec_b)
    
    vectorize[sub_vectorized, simd_width](result.size)
    
    return result

fn mul[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise multiplication with SIMD vectorization.
    
    Args:
        a: First tensor
        b: Second tensor
    
    Returns:
        Result tensor (a * b)
    """
    var result = Tensor[dtype](a.shape)
    alias simd_width = simd_width_of[dtype]()
    
    @parameter
    fn mul_vectorized[width: Int](i: Int):
        var vec_a = a.data.load[width=width](i)
        var vec_b = b.data.load[width=width](i)
        result.data.store[width=width](i, vec_a * vec_b)
    
    vectorize[mul_vectorized, simd_width](result.size)
    
    return result

fn div[dtype: DType](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    """Element-wise division with SIMD vectorization.
    
    Args:
        a: First tensor (numerator)
        b: Second tensor (denominator)
    
    Returns:
        Result tensor (a / b)
    """
    var result = Tensor[dtype](a.shape)
    alias simd_width = simd_width_of[dtype]()
    
    @parameter
    fn div_vectorized[width: Int](i: Int):
        var vec_a = a.data.load[width=width](i)
        var vec_b = b.data.load[width=width](i)
        result.data.store[width=width](i, vec_a / vec_b)
    
    vectorize[div_vectorized, simd_width](result.size)
    
    return result

# =============================================================================
# Scalar Operations
# =============================================================================

fn scalar_add[dtype: DType](tensor: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
    """Add a scalar to all elements.
    
    Args:
        tensor: Input tensor
        scalar: Scalar value to add
    
    Returns:
        Result tensor (tensor + scalar)
    """
    var result = Tensor[dtype](tensor.shape)
    alias simd_width = simd_width_of[dtype]()
    
    @parameter
    fn add_vectorized[width: Int](i: Int):
        var vec = tensor.data.load[width=width](i)
        result.data.store[width=width](i, vec + scalar)
    
    vectorize[add_vectorized, simd_width](result.size)
    
    return result

fn scalar_mul[dtype: DType](tensor: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
    """Multiply all elements by a scalar.
    
    Args:
        tensor: Input tensor
        scalar: Scalar multiplier
    
    Returns:
        Result tensor (tensor * scalar)
    """
    var result = Tensor[dtype](tensor.shape)
    alias simd_width = simd_width_of[dtype]()
    
    @parameter
    fn mul_vectorized[width: Int](i: Int):
        var vec = tensor.data.load[width=width](i)
        result.data.store[width=width](i, vec * scalar)
    
    vectorize[mul_vectorized, simd_width](result.size)
    
    return result

# =============================================================================
# Unary Operations
# =============================================================================

fn neg[dtype: DType](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Negate all elements.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Result tensor (-tensor)
    """
    return scalar_mul(tensor, -1.0)

fn abs[dtype: DType](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Absolute value of all elements.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Result tensor (|tensor|)
    """
    var result = Tensor[dtype](tensor.shape)
    alias simd_width = simd_width_of[dtype]()
    
    @parameter
    fn abs_vectorized[width: Int](i: Int):
        var vec = tensor.data.load[width=width](i)
        result.data.store[width=width](i, abs(vec))
    
    vectorize[abs_vectorized, simd_width](result.size)
    
    return result

fn exp[dtype: DType](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Exponential function (e^x) for all elements.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Result tensor (e^tensor)
    """
    var result = Tensor[dtype](tensor.shape)
    
    # Element-wise exp (no SIMD version in stdlib yet)
    for i in range(result.size):
        result.data[i] = exp(tensor.data[i])
    
    return result

fn log[dtype: DType](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Natural logarithm for all elements.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Result tensor (ln(tensor))
    """
    var result = Tensor[dtype](tensor.shape)
    
    # Element-wise log
    for i in range(result.size):
        result.data[i] = log(tensor.data[i])
    
    return result

fn sqrt[dtype: DType](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Square root for all elements.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Result tensor (√tensor)
    """
    var result = Tensor[dtype](tensor.shape)
    alias simd_width = simd_width_of[dtype]()
    
    @parameter
    fn sqrt_vectorized[width: Int](i: Int):
        var vec = tensor.data.load[width=width](i)
        result.data.store[width=width](i, sqrt(vec))
    
    vectorize[sqrt_vectorized, simd_width](result.size)
    
    return result

fn pow[dtype: DType](tensor: Tensor[dtype], exponent: Scalar[dtype]) -> Tensor[dtype]:
    """Raise all elements to a power.
    
    Args:
        tensor: Input tensor
        exponent: Power to raise to
    
    Returns:
        Result tensor (tensor^exponent)
    """
    var result = Tensor[dtype](tensor.shape)
    
    # Element-wise power
    for i in range(result.size):
        result.data[i] = pow(tensor.data[i], exponent)
    
    return result
