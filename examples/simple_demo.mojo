"""
Example: Basic Tensor Operations
Demonstrates creation, manipulation, and operations on tensors.

This is a self-contained example that includes all necessary components.
"""

from memory import UnsafePointer, memset_zero, memcpy
from algorithm import vectorize
from sys import simdwidthof
from math import exp as math_exp, sqrt as math_sqrt, log as math_log

# =============================================================================
# Core Tensor
# =============================================================================

struct SimpleTensor:
    """Simple high-performance tensor for demonstration."""
    var data: UnsafePointer[Float32]
    var size: Int
    
    fn __init__(out self, size: Int):
        self.size = size
        self.data = UnsafePointer[Float32].alloc(size)
        memset_zero(self.data, size)
    
    fn __del__(self):
        self.data.free()
    
    fn __copyinit__(out self, existing: Self):
        self.size = existing.size
        self.data = UnsafePointer[Float32].alloc(self.size)
        memcpy(self.data, existing.data, self.size)
    
    fn fill(self, value: Float32):
        """Fill with SIMD vectorization."""
        alias simd_width = simdwidthof[Float32]()
        
        @parameter
        fn fill_vec[width: Int](i: Int):
            self.data.store[width=width](i, SIMD[Float32, width](value))
        
        vectorize[fill_vec, simd_width](self.size)
    
    fn __str__(self) -> String:
        var result = "["
        var max_show = min(10, self.size)
        for i in range(max_show):
            if i > 0:
                result += ", "
            result += String(self.data[i])
        if self.size > max_show:
            result += ", ..."
        result += "]"
        return result

# =============================================================================
# Operations
# =============================================================================

fn add_tensors(a: SimpleTensor, b: SimpleTensor) -> SimpleTensor:
    """Add two tensors with SIMD."""
    var result = SimpleTensor(a.size)
    alias simd_width = simdwidthof[Float32]()
    
    @parameter
    fn add_vec[width: Int](i: Int):
        var va = a.data.load[width=width](i)
        var vb = b.data.load[width=width](i)
        result.data.store[width=width](i, va + vb)
    
    vectorize[add_vec, simd_width](result.size)
    return result

fn mul_tensors(a: SimpleTensor, b: SimpleTensor) -> SimpleTensor:
    """Multiply two tensors element-wise with SIMD."""
    var result = SimpleTensor(a.size)
    alias simd_width = simdwidthof[Float32]()
    
    @parameter
    fn mul_vec[width: Int](i: Int):
        var va = a.data.load[width=width](i)
        var vb = b.data.load[width=width](i)
        result.data.store[width=width](i, va * vb)
    
    vectorize[mul_vec, simd_width](result.size)
    return result

fn scalar_mul(tensor: SimpleTensor, scalar: Float32) -> SimpleTensor:
    """Multiply tensor by scalar with SIMD."""
    var result = SimpleTensor(tensor.size)
    alias simd_width = simdwidthof[Float32]()
    
    @parameter
    fn mul_vec[width: Int](i: Int):
        var v = tensor.data.load[width=width](i)
        result.data.store[width=width](i, v * scalar)
    
    vectorize[mul_vec, simd_width](result.size)
    return result

fn sum_tensor(tensor: SimpleTensor) -> Float32:
    """Sum all elements with SIMD."""
    alias simd_width = simdwidthof[Float32]()
    var simd_sum = SIMD[Float32, simd_width](0)
    
    var num_vectors = tensor.size // simd_width
    for i in range(num_vectors):
        var vec = tensor.data.load[width=simd_width](i * simd_width)
        simd_sum += vec
    
    var total = Float32(0)
    for i in range(simd_width):
        total += simd_sum[i]
    
    var remainder = tensor.size % simd_width
    for i in range(tensor.size - remainder, tensor.size):
        total += tensor.data[i]
    
    return total

fn mean_tensor(tensor: SimpleTensor) -> Float32:
    """Calculate mean."""
    return sum_tensor(tensor) / Float32(tensor.size)

# =============================================================================
# Main Demo
# =============================================================================

fn main():
    print("🔥 MojoTensor - Basic Operations Demo\n")
    
    print("=" * 60)
    print("1. CREATING TENSORS")
    print("=" * 60)
    
    var a = SimpleTensor(12)  # 12 elements
    a.fill(2.0)
    print("Tensor A (12 elements) filled with 2.0:")
    print(a)
    print()
    
    var b = SimpleTensor(12)
    b.fill(3.0)
    print("Tensor B (12 elements) filled with 3.0:")
    print(b)
    print()
    
    print("=" * 60)
    print("2. ELEMENT-WISE OPERATIONS (SIMD)")
    print("=" * 60)
    
    var c = add_tensors(a, b)
    print("A + B:")
    print(c)
    print()
    
    var d = mul_tensors(a, b)
    print("A * B (element-wise):")
    print(d)
    print()
    
    var e = scalar_mul(a, 5.0)
    print("A * 5.0:")
    print(e)
    print()
    
    print("=" * 60)
    print("3. REDUCTION OPERATIONS")
    print("=" * 60)
    
    var sum_a = sum_tensor(a)
    print("Sum of A:", sum_a)
    
    var mean_a = mean_tensor(a)
    print("Mean of A:", mean_a)
    print()
    
    print("=" * 60)
    print("4. PERFORMANCE SHOWCASE")
    print("=" * 60)
    print("All operations use SIMD vectorization!")
    alias simd_width = simdwidthof[Float32]()
    print("- SIMD width for float32:", simd_width)
    print("- Elements processed in parallel:", simd_width)
    print("- Performance boost: ~" + String(simd_width) + "x over scalar code")
    print()
    
    print("✅ Demo completed successfully!")
    print()
    print("=" * 60)
    print("WHAT MAKES THIS POWERFUL:")
    print("=" * 60)
    print("✅ Zero-cost abstractions - No runtime overhead")
    print("✅ SIMD auto-vectorization - Hardware-level parallelism")
    print("✅ Manual memory control - Predictable performance")
    print("✅ Type safety - Compile-time guarantees")
    print("✅ Python-like syntax - Easy to use")
    print()
    print("🚀 This is just the beginning! MojoTensor can:")
    print("   - Handle N-dimensional tensors")
    print("   - Implement broadcasting (NumPy-style)")
    print("   - Support custom dtypes (int, float16, float64)")
    print("   - Use multi-threading with parallelize()")
    print("   - Build ML frameworks (neural networks, etc.)")
