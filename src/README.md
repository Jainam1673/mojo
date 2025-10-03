# 🔥 MojoTensor Core Library

This directory contains the core implementation of the MojoTensor library.

## ✅ Status: Updated to Mojo 0.25.6

All files have been updated with the latest Mojo syntax and API:
- ✅ `simd_width_of` instead of deprecated `simdwidthof`
- ✅ `out self` for constructors (__init__)
- ✅ `__del__(self)` without `owned` keyword
- ✅ `__copyinit__(out self, existing: Self)` with correct parameters
- ✅ Proper `inout self` for mutating methods

## 📁 Structure

```
src/
├── core/                    # Core tensor fundamentals
│   ├── __init__.mojo       # Package initialization
│   ├── dtype.mojo          # Data type system (✅ updated)
│   ├── shape.mojo          # Shape and stride management (✅ updated)
│   └── tensor.mojo         # N-D tensor implementation (✅ updated)
│
├── ops/                     # Operations
│   ├── __init__.mojo       # Package initialization
│   ├── elementwise.mojo    # Element-wise operations (✅ updated)
│   ├── reduction.mojo      # Reductions (sum, mean, etc.) (✅ updated)
│   └── linalg.mojo         # Linear algebra (✅ updated)
│
├── __init__.mojo           # Main package initialization
└── test_syntax.mojo        # Syntax verification test
```

## 🎯 Features Implemented

### Core Types
- **Tensor[dtype: DType]** - Generic N-D tensor with SIMD optimization
- **TensorShape** - Shape and stride management for multi-dimensional arrays
- **DType System** - Type-safe data type abstractions

### Operations
**Element-wise (SIMD optimized):**
- Basic: add, sub, mul, div
- Scalar: scalar_add, scalar_mul
- Unary: neg, abs, exp, log, sqrt, pow

**Reductions (SIMD optimized):**
- sum, mean, variance, std
- max, min, argmax, argmin
- prod (product)

**Linear Algebra:**
- matmul, matmul_tiled (matrix multiplication)
- dot (dot product)
- outer (outer product)
- trace, norm

## 🚧 Current Limitation: Module System

**Note:** The Mojo module/package system is still evolving. Currently, importing from `src/` in external files may not work as expected.

**Workaround:** For now, use self-contained examples (see `examples/working_demo.mojo`) that include all necessary code inline.

## 🔧 Using the Library

### Option 1: Self-Contained Code (Recommended for now)
Copy the patterns from `examples/working_demo.mojo`:
```mojo
from sys import simd_width_of
from algorithm import vectorize
from memory import UnsafePointer

# Inline your tensor operations...
```

### Option 2: Direct File Usage (When package system matures)
Once Mojo's package system is more stable:
```mojo
from src.core.tensor import Tensor
from src.ops.elementwise import add, mul

var a = Tensor[DType.float32](10, 10)
var b = Tensor[DType.float32](10, 10)
var c = add(a, b)
```

## 📝 Code Quality

All source files follow best practices:
- ✅ **Type Safety** - Generic types with compile-time checks
- ✅ **Documentation** - Comprehensive docstrings
- ✅ **SIMD Optimization** - Vectorized operations throughout
- ✅ **Memory Safety** - Explicit ownership and lifetimes
- ✅ **Latest Syntax** - Mojo 0.25.6 compatible

## 🧪 Testing

Run the syntax verification:
```bash
pixi run mojo src/test_syntax.mojo
```

This verifies that all updated syntax compiles correctly.

## 📚 Key Design Decisions

### 1. Generic Tensor Type
```mojo
struct Tensor[dtype: DType]:
    var shape: TensorShape
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var owns_data: Bool
```

### 2. SIMD Everywhere
All operations use SIMD vectorization:
```mojo
@parameter
fn operation[width: Int](i: Int):
    var vec = data.load[width=width](i)
    # Process 'width' elements at once

vectorize[operation, simd_width](size)
```

### 3. Zero-Copy Views
Tensors can be views over existing data:
```mojo
var view = Tensor[dtype](shape, existing_data, owns_data=False)
```

### 4. Compile-Time Optimization
Using `@parameter` and `@always_inline` for maximum performance:
```mojo
@always_inline
fn get_simd_width[dtype: DType]() -> Int:
    return simd_width_of[dtype]()
```

## 🚀 Next Steps

1. **Fix Module System** - Work with Mojo team or wait for package system improvements
2. **Add Tests** - Comprehensive unit tests for all operations
3. **Benchmarks** - Performance comparisons with NumPy/PyTorch
4. **Advanced Features**:
   - Broadcasting implementation
   - Memory pooling
   - Parallel execution with `parallelize()`
   - GPU support

## 🔗 Related Files

- **Examples:** See `examples/working_demo.mojo` for working code
- **Documentation:** See `ROADMAP.md` for development plan
- **Quick Start:** See `QUICKSTART.md` for setup instructions

## 💡 Contributing

When updating code:
1. Use `simd_width_of` not `simdwidthof`
2. Use `out self` for __init__
3. Use `__del__(self)` without `owned`
4. Use `__copyinit__(out self, existing: Self)`
5. Test with `pixi run mojo your_file.mojo`

## 📖 Learning Resources

- **Mojo Docs:** https://docs.modular.com/mojo/
- **SIMD Programming:** https://docs.modular.com/mojo/manual/vectors
- **Memory Management:** https://docs.modular.com/mojo/manual/pointers

---

**Status:** Core library syntax updated ✅  
**Mojo Version:** 0.25.6  
**Last Updated:** October 3, 2025
