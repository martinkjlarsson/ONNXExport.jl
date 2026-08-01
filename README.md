# ONNXExport.jl

[![Build Status](https://github.com/martinkjlarsson/ONNXExport.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/martinkjlarsson/ONNXExport.jl/actions/workflows/CI.yml?query=branch%3Amain)

**This package is under development and is not registered. Support is limited and likely unstable. Feel free to file issues.**

---

ONNXExport traces Julia functions and exports them as ONNX models. The package cannot import or run models.

Install using
```julia
] add https://github.com/martinkjlarsson/ONNXExport.jl.git
```

# Examples
The exported models can be inspected using [netron](https://netron.app/).

```julia
using ONNXExport

f(x, y) = x .+ y .- 3
ONNXExport.save("model.onnx", f, rand(Float32, 3, 4), rand(Float32, 3))
```

```julia
using ONNXExport, Lux, Random

model = Chain(Dense(16 => 8, relu), Dense(8 => 2))

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)
st = Lux.testmode(st)

f(x) = first(Lux.apply(model, x, ps, st))
ONNXExport.save("model.onnx", f, ProbeArray{Float32}("input1", 16, :N))
```

# Design
ONNXExport works by defining a custom `AbstractArray` subtype `ProbeArray` and `Number` subtype `ProbeNumber`. These can be passed to many functions that normally accept `AbstractArray`s or `Number`s, e.g., arithmetic and linear algebra operations, neural networks, and array manipulations. Instead of performing the operations, the `Probe*` types write operators to an ONNX graph, which can later be saved to file. This enables the tracing of a Julia function with export to ONNX.

Broadcasting is also supported through the `BroadcastProbe <: Number` type. It wraps a `ProbeArray` such that the array can be passed to functions accepting `Number`s, and any operation performed is replaced with elementwise ONNX operators.

# Support
The focus of the package has been to export models from [Lux.jl](https://lux.csail.mit.edu/stable/), but much more work is needed to support all types of layers. See [Julia functions](docs/support_julia.md), [ONNX operators](docs/support_onnx.md), and [Lux layers](docs/support_lux.md) for details. The internal and external APIs are not yet stable. There are also several limitations listed below, some of which might be solved in future versions.

## Limitations
ONNXExport cannot convert any arbitrary Julia function into ONNX, partly because of limitations in ONNXExport, but also due to limitations in ONNX itself.

### Symbolic Dimensions
ONNXExport supports symbolic dimensions, dimensions with unknown size at export, but this has not been fully tested and will likely fail for many functions.

### Control Flow
Due to the way ONNXExport traces Julia functions, it is not possible to capture certain control flow statements such as `if`, `for`, and `while`. These will likely result in the error `TypeError: non-boolean (ProbeNumber{Bool}) used in boolean context`. Try instead to use array operations or `ifelse`.

### Random
Random numbers from `rand`, `randn`, `randexp`, and `bitrand` are supported if the `ProbeRNG` generator is used.
```julia
using ONNXExport

rng = ProbeRNG()

f(x) = x * rand(rng, Float32, 4)
ONNXExport.save("model.onnx", f, rand(Float32, 3, 4))
```

### Broadcasting
Broadcasting support is limited to element-wise operations, e.g., `sin.(A)` and `relu.(Wx .+ b)`. Nested broadcasting or broadcasting over slices will likely fail.

### Immutability
ONNX tensors are immutable. Consequently, ONNXExport does not support mutating functions such as `setindex!`.

### Varargs Functions
The tracing works by overloading commonly used functions with methods taking `ProbeArray` arguments instead of `AbstractArray`s. This poses problems for functions such as `cat`, which can accept an arbitrary number of arguments. The tracing might fail if none of the first few arguments are of type `ProbeArray`.

### Small Integers and ONNX Runtime
The [ONNX documentation](https://onnx.ai/onnx/index.html) specifies that many operators, such as [Add](https://onnx.ai/onnx/operators/onnx__Add.html), support all real tensor data types, e.g., `int8` and `uint16`. However, this does not mean the various ONNX runtimes do (see [this issue](https://github.com/microsoft/onnxruntime/issues/19231)). A successful export does not guarantee a runnable model. `Float32`, `Int32`, and `Int64` are typically safe types to use.

### Complex Numbers
Complex numbers are currently not supported. Although the ONNX specification defines the `complex64` and `complex128` tensor data types, corresponding to the Julia types `ComplexF32` and `ComplexF64`, respectively, there are no operators that support them. The existing operators that use complex numbers, e.g., [DFT](https://onnx.ai/onnx/operators/onnx__DFT.html) and [ComplexMul](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftcomplexmul), use an interleaved representation, where the fastest changing dimension have size 2, corresponding to the real and imaginary parts. Conversion between complex Julia types and this array representation can be done with [reinterpret](https://docs.julialang.org/en/v1/base/arrays/#Base.reinterpret):
```julia
julia> size(reinterpret(reshape, Float32, rand(ComplexF32, 3, 4)))
(2, 3, 4)
```

### Strings
ONNX supports UTF-8 encoded strings with some operator support ([RegexFullMatch](https://onnx.ai/onnx/operators/onnx__RegexFullMatch.html), [StringConcat](https://onnx.ai/onnx/operators/onnx__StringConcat.html), [StringNormalizer](https://onnx.ai/onnx/operators/onnx__StringNormalizer.html), etc). However, ONNXExport does currently not support strings.

### Large Models and External Data
ONNX protobuf files are limited to 2GB in size. Larger models need to store tensor data externally alongside the ONNX file. This is currently not supported, and consequently, exported models are limited to 2GB.

# Indexing in Julia and ONNX
Julia uses [column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order) (like Fortran and MATLAB), while ONNX uses row-major order (like C and Python/NumPy). In Julia, the leftmost index varies fastest, while in ONNX, the rightmost index varies fastest. To solve this discrepancy, we reverse the dimensions when writing Julia arrays as ONNX tensors. A Julia array of size `(row, column, batch)` is written to ONNX as a tensor of shape `(batch, column, row)`. Note that the data remains unchanged, e.g., `row` is the fastest varying dimension in both cases.

However, ONNX expects tensors in the form `(batch, row, column)` for matrix operations, so we have effectively transposed the rows and columns. The solution to this is to transpose all matrix operations.
For example, a linear layer might perform the operation
```math
y = Wx + b.
```
When converted to ONNX, this becomes
```math
y^T = x^T W^T + b^T,
```
where the transposes come for free due to the reversal of the dimensions. Importantly, we must switch the argument order in the matrix multiplication: $Wx \rightarrow x^T W^T$.

Furthermore, Julia uses one-based indexing, while ONNX uses zero-based indexing. The conversion is trivial in most cases, but care is required when indexing dimensions. Dimension `d` in Julia corresponds to dimension `ndims(A) - d` in ONNX. For example, `permutedims(A, (2, 1, 3, 4))` in Julia, i.e., permuting the two fastest varying/leftmost dimensions, will be written to ONNX as a [Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html) operator with the `perm` attribute `(0, 1, 3, 2)`, i.e., permuting the two fastest varying/rightmost dimensions.
