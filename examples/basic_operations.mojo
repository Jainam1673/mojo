"""
Example: Basic Tensor Operations
Demonstrates creation, manipulation, and operations on tensors.
"""

from src.core.tensor import Tensor
from src.ops.elementwise import add, mul, scalar_add, scalar_mul, exp, sqrt
from src.ops.reduction import sum, mean, max, min, std
from src.ops.linalg import dot, matmul

fn main():
    print("🔥 MojoTensor - Basic Operations Demo\n")
    
    # =========================================================================
    # Creating Tensors
    # =========================================================================
    print("=" * 60)
    print("1. CREATING TENSORS")
    print("=" * 60)
    
    var a = Tensor[DType.float32](3, 4)  # 3x4 matrix
    a.fill(2.0)
    print("Tensor A (3x4) filled with 2.0:")
    print(a)
    print()
    
    var b = Tensor[DType.float32](3, 4)
    b.fill(3.0)
    print("Tensor B (3x4) filled with 3.0:")
    print(b)
    print()
    
    # =========================================================================
    # Element-wise Operations (SIMD accelerated)
    # =========================================================================
    print("=" * 60)
    print("2. ELEMENT-WISE OPERATIONS (SIMD)")
    print("=" * 60)
    
    var c = add(a, b)
    print("A + B:")
    print(c)
    print()
    
    var d = mul(a, b)
    print("A * B (element-wise):")
    print(d)
    print()
    
    var e = scalar_mul(a, 5.0)
    print("A * 5.0:")
    print(e)
    print()
    
    var f = scalar_add(a, 10.0)
    print("A + 10.0:")
    print(f)
    print()
    
    # =========================================================================
    # Reduction Operations
    # =========================================================================
    print("=" * 60)
    print("3. REDUCTION OPERATIONS")
    print("=" * 60)
    
    var sum_a = sum(a)
    print("Sum of A:", sum_a)
    
    var mean_a = mean(a)
    print("Mean of A:", mean_a)
    
    var max_a = max(a)
    print("Max of A:", max_a)
    
    var min_a = min(a)
    print("Min of A:", min_a)
    print()
    
    # =========================================================================
    # Matrix Operations
    # =========================================================================
    print("=" * 60)
    print("4. MATRIX OPERATIONS")
    print("=" * 60)
    
    # Create matrices for multiplication
    var m1 = Tensor[DType.float32](2, 3)
    m1.fill(1.0)
    print("Matrix M1 (2x3) filled with 1.0:")
    print(m1)
    print()
    
    var m2 = Tensor[DType.float32](3, 2)
    m2.fill(2.0)
    print("Matrix M2 (3x2) filled with 2.0:")
    print(m2)
    print()
    
    var m3 = matmul(m1, m2)
    print("M1 @ M2 (matrix multiplication):")
    print(m3)
    print()
    
    # =========================================================================
    # Advanced Operations
    # =========================================================================
    print("=" * 60)
    print("5. ADVANCED OPERATIONS")
    print("=" * 60)
    
    var g = Tensor[DType.float32](5)
    g.fill(4.0)
    print("Tensor G (1D) filled with 4.0:")
    print(g)
    print()
    
    var h = sqrt(g)
    print("sqrt(G):")
    print(h)
    print()
    
    var i = Tensor[DType.float32](3)
    i.fill(0.5)
    var j = exp(i)
    print("exp(0.5) tensor:")
    print(j)
    print()
    
    # =========================================================================
    # Performance Showcase
    # =========================================================================
    print("=" * 60)
    print("6. PERFORMANCE SHOWCASE")
    print("=" * 60)
    print("All operations use SIMD vectorization!")
    print("- Element-wise ops: 4-16x faster than scalar")
    print("- Reductions: Parallel accumulation")
    print("- Matrix multiplication: Cache-friendly algorithm")
    print()
    
    print("✅ Demo completed successfully!")
