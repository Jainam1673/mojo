"""
Core Tensor Implementation
High-performance N-dimensional array with SIMD vectorizat        self.shape = existing.shape
        self.size = existing.size
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self.owns_data = True
        
        # Copy data
        memcpy(self.data, existing.data, self.size) zero-copy operations.
"""

from memory import UnsafePointer, memset_zero, memcpy
from algorithm import vectorize, parallelize
from sys import simd_width_of
from collections import DynamicVector
from .shape import TensorShape
from .dtype import get_simd_width, dtype_size, dtype_name

struct Tensor[dtype: DType](Movable):
    """High-performance N-dimensional tensor with SIMD optimization.
    
    Key Features:
    - SIMD vectorized operations for maximum performance
    - Zero-copy views and slicing where possible
    - Automatic broadcasting for element-wise operations
    - Cache-friendly memory layout
    - Type-safe generic implementation
    
    Parameters:
        dtype: The data type of tensor elements (float32, int64, etc.)
    
    Attributes:
        shape: Shape and stride information
        data: Raw pointer to the data buffer
        size: Total number of elements
        owns_data: Whether this tensor owns its data (for cleanup)
    """
    var shape: TensorShape
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var owns_data: Bool
    
    fn __init__(out self, *dims: Int):
        """Create a new tensor with the given dimensions.
        
        Allocates memory and initializes to zero.
        
        Args:
            dims: Dimension sizes (e.g., 3, 4, 5 for 3x4x5 tensor)
        """
        self.shape = TensorShape(dims)
        self.size = self.shape.num_elements()
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self.owns_data = True
        
        # Initialize to zero
        memset_zero(self.data, self.size)
    
    fn __init__(out self, shape: TensorShape):
        """Create a new tensor with the given shape.
        
        Args:
            shape: The shape of the tensor
        """
        self.shape = shape
        self.size = self.shape.num_elements()
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self.owns_data = True
        
        memset_zero(self.data, self.size)
    
    fn __init__(out self, shape: TensorShape, data: UnsafePointer[Scalar[dtype]], owns_data: Bool = False):
        """Create a tensor view over existing data (zero-copy).
        
        Args:
            shape: The shape of the tensor
            data: Pointer to existing data
            owns_data: Whether this tensor should free the data on destruction
        """
        self.shape = shape
        self.size = self.shape.num_elements()
        self.data = data
        self.owns_data = owns_data
    
    fn __deinit__(var self):
        """Free memory if this tensor owns its data."""
        if self.owns_data:
            self.data.free()
    
    fn __copyinit__(out self, existing: Self):
        """Deep copy constructor.
        
        Args:
            existing: The tensor to copy from
        """
        self.shape = existing.shape
        self.size = existing.size
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self.owns_data = True
        
        # Copy data
        memcpy(self.data, existing.data, self.size)
    
    @always_inline
    fn __getitem__(self, indices: DynamicVector[Int]) -> Scalar[dtype]:
        """Get element at the given multi-dimensional index.
        
        Args:
            indices: Index for each dimension
        
        Returns:
            The element at the specified position.
        """
        var offset = self.shape.linear_index(indices)
        return self.data[offset]
    
    @always_inline
    fn __setitem__(inout self, indices: DynamicVector[Int], value: Scalar[dtype]):
        """Set element at the given multi-dimensional index.
        
        Args:
            indices: Index for each dimension
            value: Value to set
        """
        var offset = self.shape.linear_index(indices)
        self.data[offset] = value
    
    fn fill(inout self, value: Scalar[dtype]):
        """Fill the entire tensor with a scalar value.
        
        Uses SIMD for maximum performance.
        
        Args:
            value: Value to fill with
        """
        alias simd_width = simd_width_of[dtype]()
        
        @parameter
        fn fill_vectorized[width: Int](i: Int):
            self.data.store[width=width](i, SIMD[dtype, width](value))
        
        vectorize[fill_vectorized, simd_width](self.size)
    
    fn zeros(inout self):
        """Fill tensor with zeros."""
        memset_zero(self.data, self.size)
    
    fn ones(inout self):
        """Fill tensor with ones."""
        self.fill(1.0)
    
    fn __str__(self) -> String:
        """String representation of the tensor.
        
        Returns:
            String showing shape, dtype, and sample of data.
        """
        var result = "Tensor" + String(self.shape) + " dtype=" + dtype_name[dtype]() + "\n["
        
        # Show first few elements
        var max_show = min(10, self.size)
        for i in range(max_show):
            if i > 0:
                result += ", "
            result += String(self.data[i])
        
        if self.size > max_show:
            result += ", ..."
        
        result += "]"
        return result
    
    fn reshape(self, *new_dims: Int) -> Self:
        """Reshape tensor to new dimensions (zero-copy view if possible).
        
        Args:
            new_dims: New dimension sizes
        
        Returns:
            Reshaped tensor (view of the same data).
        """
        var new_shape = TensorShape(new_dims)
        
        # Verify same total size
        # TODO: Add proper error handling
        var new_size = new_shape.num_elements()
        
        # Return view with new shape
        return Tensor[dtype](new_shape, self.data, False)
    
    fn transpose(self) -> Self:
        """Transpose 2D tensor (swap dimensions).
        
        For 2D tensors only. Creates a new tensor with transposed data.
        
        Returns:
            Transposed tensor.
        """
        # For now, simple implementation for 2D tensors
        # TODO: Handle n-dimensional transpose
        
        var rows = self.shape.dims[0]
        var cols = self.shape.dims[1]
        var result = Tensor[dtype](cols, rows)
        
        for i in range(rows):
            for j in range(cols):
                var src_idx = DynamicVector[Int]()
                src_idx.push_back(i)
                src_idx.push_back(j)
                
                var dst_idx = DynamicVector[Int]()
                dst_idx.push_back(j)
                dst_idx.push_back(i)
                
                result[dst_idx] = self[src_idx]
        
        return result
    
    @always_inline
    fn num_elements(self) -> Int:
        """Get total number of elements."""
        return self.size
    
    @always_inline
    fn rank(self) -> Int:
        """Get number of dimensions."""
        return self.shape.rank

# Alias for convenience
alias T = Tensor
