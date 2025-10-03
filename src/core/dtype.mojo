"""
Data Type System for MojoTensor
Provides a flexible, type-safe system for numeric computations.
"""

from sys import simd_width_of

# Type aliases for common numeric types
alias Float16 = DType.float16
alias Float32 = DType.float32
alias Float64 = DType.float64
alias Int8 = DType.int8
alias Int16 = DType.int16
alias Int32 = DType.int32
alias Int64 = DType.int64
alias UInt8 = DType.uint8
alias UInt16 = DType.uint16
alias UInt32 = DType.uint32
alias UInt64 = DType.uint64

@always_inline
fn get_simd_width[dtype: DType]() -> Int:
    """Get optimal SIMD width for the given dtype.
    
    Returns the number of elements that can be processed in parallel
    using SIMD instructions for maximum hardware utilization.
    
    Parameters:
        dtype: The data type to get SIMD width for.
    
    Returns:
        Number of elements processable in one SIMD operation.
    """
    return simd_width_of[dtype]()

@always_inline
fn dtype_size[dtype: DType]() -> Int:
    """Get the size in bytes of a dtype.
    
    Parameters:
        dtype: The data type to get size for.
    
    Returns:
        Size in bytes.
    """
    if dtype == DType.float64 or dtype == DType.int64 or dtype == DType.uint64:
        return 8
    elif dtype == DType.float32 or dtype == DType.int32 or dtype == DType.uint32:
        return 4
    elif dtype == DType.float16 or dtype == DType.int16 or dtype == DType.uint16:
        return 2
    else:  # int8, uint8
        return 1

@always_inline
fn dtype_name[dtype: DType]() -> String:
    """Get human-readable name for dtype.
    
    Parameters:
        dtype: The data type.
    
    Returns:
        String representation of the dtype.
    """
    if dtype == DType.float64:
        return "float64"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.float16:
        return "float16"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.uint64:
        return "uint64"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint16:
        return "uint16"
    else:  # uint8
        return "uint8"

@always_inline
fn is_floating_point[dtype: DType]() -> Bool:
    """Check if dtype is a floating point type.
    
    Parameters:
        dtype: The data type to check.
    
    Returns:
        True if floating point, False otherwise.
    """
    return dtype == DType.float16 or dtype == DType.float32 or dtype == DType.float64

@always_inline
fn is_integer[dtype: DType]() -> Bool:
    """Check if dtype is an integer type.
    
    Parameters:
        dtype: The data type to check.
    
    Returns:
        True if integer, False otherwise.
    """
    return not is_floating_point[dtype]()
