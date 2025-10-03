"""
MojoTensor Simple Demo - Working example with Mojo 0.25.6
Demonstrates SIMD vectorization and high-performance computing.
"""

from sys import simd_width_of
from algorithm import vectorize
from memory import UnsafePointer, memset_zero

fn main():
    print("🔥 MojoTensor - High-Performance Computing Demo\n")
    
    # =========================================================================
    # SIMD-Accelerated Vector Addition
    # =========================================================================
    print("=" * 60)
    print("1. SIMD-ACCELERATED VECTOR OPERATIONS")
    print("=" * 60)
    
    alias size = 1024
    alias simd_width = simd_width_of[Float32]()
    
    # Allocate memory
    var a = UnsafePointer[Float32].alloc(size)
    var b = UnsafePointer[Float32].alloc(size)
    var result = UnsafePointer[Float32].alloc(size)
    
    # Initialize data
    for i in range(size):
        a[i] = Float32(i)
        b[i] = Float32(i * 2)
    
    # SIMD vectorized addition
    @parameter
    fn add_vectorized[width: Int](i: Int):
        var vec_a = a.load[width=width](i)
        var vec_b = b.load[width=width](i)
        result.store[width=width](i, vec_a + vec_b)
    
    vectorize[add_vectorized, simd_width](size)
    
    print("✅ Added", size, "elements using SIMD")
    print("   SIMD width:", simd_width, "elements/instruction")
    print("   Speedup: ~", simd_width, "x over scalar code")
    print("   First few results:", result[0], result[1], result[2], result[3])
    print()
    
    # =========================================================================
    # SIMD Reductions
    # =========================================================================
    print("=" * 60)
    print("2. SIMD PARALLEL REDUCTION")
    print("=" * 60)
    
    var simd_sum = SIMD[DType.float32, simd_width](0)
    var num_vectors = size // simd_width
    
    for i in range(num_vectors):
        var vec = result.load[width=simd_width](i * simd_width)
        simd_sum += vec
    
    var total = Float32(0)
    for i in range(simd_width):
        total += simd_sum[i]
    
    # Add remainder
    var remainder = size % simd_width
    for i in range(size - remainder, size):
        total += result[i]
    
    print("✅ Sum of all", size, "elements:", total)
    print("   Using parallel SIMD accumulation")
    print()
    
    # =========================================================================
    # Element-wise Multiplication (Hadamard Product)
    # =========================================================================
    print("=" * 60)
    print("3. ELEMENT-WISE MULTIPLICATION")
    print("=" * 60)
    
    var product = UnsafePointer[Float32].alloc(size)
    
    @parameter
    fn mul_vectorized[width: Int](i: Int):
        var vec_a = a.load[width=width](i)
        var vec_b = b.load[width=width](i)
        product.store[width=width](i, vec_a * vec_b)
    
    vectorize[mul_vectorized, simd_width](size)
    
    print("✅ Multiplied", size, "elements")
    print("   Result[0] =", product[0], "(0 * 0)")
    print("   Result[1] =", product[1], "(1 * 2)")
    print("   Result[2] =", product[2], "(2 * 4)")
    print()
    
    # =========================================================================
    # Scalar Operations
    # =========================================================================
    print("=" * 60)
    print("4. BROADCAST SCALAR OPERATIONS")
    print("=" * 60)
    
    var scaled = UnsafePointer[Float32].alloc(size)
    var scalar = Float32(3.14)
    
    @parameter
    fn scale_vectorized[width: Int](i: Int):
        var vec = a.load[width=width](i)
        scaled.store[width=width](i, vec * scalar)
    
    vectorize[scale_vectorized, simd_width](size)
    
    print("✅ Multiplied", size, "elements by", scalar)
    print("   Original[10] =", a[10])
    print("   Scaled[10] =", scaled[10])
    print()
    
    # =========================================================================
    # Performance Summary
    # =========================================================================
    print("=" * 60)
    print("🚀 PERFORMANCE HIGHLIGHTS")
    print("=" * 60)
    print()
    print("Mojo's Superpowers Demonstrated:")
    print("  ✅ SIMD Vectorization - Process", simd_width, "elements at once")
    print("  ✅ Zero-Cost Abstractions - No runtime overhead")
    print("  ✅ Manual Memory Control - Predictable, fast allocations")
    print("  ✅ Compile-Time Optimization - Parametric functions")
    print("  ✅ Hardware-Level Performance - Direct CPU feature access")
    print()
    print("What We Can Build:")
    print("  🔥 ML Frameworks - Neural networks, autograd")
    print("  🔥 Scientific Computing - Linear algebra, FFT")
    print("  🔥 Data Processing - High-throughput pipelines")
    print("  🔥 Graphics/Games - Image processing, physics")
    print("  🔥 Finance - Real-time analytics, risk models")
    print()
    print("=" * 60)
    print("This is just 1% of Mojo's power!")
    print("Ready to build production-grade libraries!")
    print("=" * 60)
    
    # Cleanup
    a.free()
    b.free()
    result.free()
    product.free()
    scaled.free()
