"""
Shape and Stride Management for Tensors
Handles multi-dimensional indexing, broadcasting, and memory layout optimization.
"""

from memory import memset_zero, memcpy
from algorithm import vectorize

struct TensorShape:
    """Represents the shape and stride information of a tensor.
    
    Attributes:
        rank: Number of dimensions
        dims: Dimension sizes
        strides: Stride for each dimension (for row-major layout)
    """
    var rank: Int
    var dims: DynamicVector[Int]
    var strides: DynamicVector[Int]
    
    fn __init__(out self, *shape: Int):
        """Initialize shape from variadic dimensions.
        
        Args:
            shape: Dimension sizes (e.g., 3, 4, 5 for 3x4x5 tensor)
        """
        self.rank = len(shape)
        self.dims = DynamicVector[Int](capacity=self.rank)
        self.strides = DynamicVector[Int](capacity=self.rank)
        
        # Store dimensions
        for i in range(self.rank):
            self.dims.push_back(shape[i])
        
        # Calculate strides (row-major/C-order)
        self._compute_strides()
    
    fn __init__(out self, shape: DynamicVector[Int]):
        """Initialize shape from a vector of dimensions.
        
        Args:
            shape: Vector of dimension sizes
        """
        self.rank = len(shape)
        self.dims = DynamicVector[Int](capacity=self.rank)
        self.strides = DynamicVector[Int](capacity=self.rank)
        
        for i in range(self.rank):
            self.dims.push_back(shape[i])
        
        self._compute_strides()
    
    fn _compute_strides(inout self):
        """Compute strides for row-major (C-order) layout.
        
        For a 3x4x5 tensor:
        - stride[2] = 1 (innermost dimension)
        - stride[1] = 5
        - stride[0] = 4 * 5 = 20
        """
        if self.rank == 0:
            return
        
        var stride = 1
        for i in range(self.rank - 1, -1, -1):
            self.strides.push_back(stride)
            stride *= self.dims[i]
        
        # Reverse strides to match dimension order
        for i in range(self.rank // 2):
            var temp = self.strides[i]
            self.strides[i] = self.strides[self.rank - 1 - i]
            self.strides[self.rank - 1 - i] = temp
    
    fn num_elements(self) -> Int:
        """Calculate total number of elements in the tensor.
        
        Returns:
            Product of all dimensions.
        """
        if self.rank == 0:
            return 0
        
        var total = 1
        for i in range(self.rank):
            total *= self.dims[i]
        return total
    
    fn linear_index(self, indices: DynamicVector[Int]) -> Int:
        """Convert multi-dimensional indices to linear index.
        
        Args:
            indices: Index for each dimension
        
        Returns:
            Linear offset in the flat array.
        """
        var offset = 0
        for i in range(self.rank):
            offset += indices[i] * self.strides[i]
        return offset
    
    fn can_broadcast(self, other: Self) -> Bool:
        """Check if two shapes are broadcast-compatible.
        
        Broadcasting rules (NumPy-style):
        - Dimensions are compatible if they're equal or one is 1
        - Shapes are aligned from the right
        
        Args:
            other: The other shape to check compatibility with
        
        Returns:
            True if shapes can be broadcast together.
        """
        var max_rank = max(self.rank, other.rank)
        
        for i in range(max_rank):
            var dim1 = self._get_dim_from_right(i)
            var dim2 = other._get_dim_from_right(i)
            
            if dim1 != dim2 and dim1 != 1 and dim2 != 1:
                return False
        
        return True
    
    fn _get_dim_from_right(self, index: Int) -> Int:
        """Get dimension counting from the right (for broadcasting).
        
        Args:
            index: Index from the right (0 is rightmost)
        
        Returns:
            Dimension size, or 1 if index is out of bounds.
        """
        var actual_index = self.rank - 1 - index
        if actual_index < 0:
            return 1
        return self.dims[actual_index]
    
    fn __eq__(self, other: Self) -> Bool:
        """Check if two shapes are equal.
        
        Args:
            other: The other shape to compare with
        
        Returns:
            True if shapes are identical.
        """
        if self.rank != other.rank:
            return False
        
        for i in range(self.rank):
            if self.dims[i] != other.dims[i]:
                return False
        
        return True
    
    fn __str__(self) -> String:
        """String representation of the shape.
        
        Returns:
            String like "(3, 4, 5)"
        """
        var result = "("
        for i in range(self.rank):
            if i > 0:
                result += ", "
            result += String(self.dims[i])
        result += ")"
        return result

@always_inline
fn broadcast_shapes(shape1: TensorShape, shape2: TensorShape) -> TensorShape:
    """Compute the broadcasted shape of two tensor shapes.
    
    Args:
        shape1: First shape
        shape2: Second shape
    
    Returns:
        The resulting broadcasted shape.
    """
    var max_rank = max(shape1.rank, shape2.rank)
    var result_dims = DynamicVector[Int](capacity=max_rank)
    
    for i in range(max_rank):
        var dim1 = shape1._get_dim_from_right(i)
        var dim2 = shape2._get_dim_from_right(i)
        result_dims.push_back(max(dim1, dim2))
    
    # Reverse to get correct order
    for i in range(max_rank // 2):
        var temp = result_dims[i]
        result_dims[i] = result_dims[max_rank - 1 - i]
        result_dims[max_rank - 1 - i] = temp
    
    return TensorShape(result_dims)
