# ONNXExport.jl

[![Build Status](https://github.com/martinkjlarsson/ONNXExport.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/martinkjlarsson/ONNXExport.jl/actions/workflows/CI.yml?query=branch%3Amain)

**This package is under development and is not registered. Support is limited and likely unstable. Feel free to file issues.**

---

ONNXExport traces Julia functions and exports them as ONNX models. The package cannot import or run models.

A Julia function `f(inputs...)` can be exported to ONNX using
```julia
export_model("model.onnx", f, inputs...)
```
where `inputs...` are example arguments from which types and sizes will be inferred.

## Examples
The exported models can be inspected using [netron](https://netron.app/).

```julia
using ONNXExport

f(x, y) = x .+ y .- 3
export_model("model.onnx", f, rand(Float32, 3, 4), rand(Float32, 3))
```

```julia
using ONNXExport, Lux, Random

model = Chain(Dense(16 => 8, relu), Dense(8 => 2))

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

f(x) = first(Lux.apply(model, x, ps, st))
export_model("model.onnx", f, ProbeArray{Float32}("input1", 16, :N))
```

# Design
ONNXExport works by defining a custom `AbstractArray` subtype `ProbeArray` and `Number` subtype `ProbeNumber`. These can be passed to many functions that normally accept `AbstractArray`s or `Number`s, e.g., arithmetic and linear algebra operations, neural networks, and array manipulations. Instead of performing the operations, the `Probe*` types write operators to an ONNX graph, which can later be saved to file. This enables the tracing of a Julia function with export to ONNX.

Broadcasting is also supported through the `BroadcastProbe <: Number` type. It wraps a `ProbeArray` such that the array can be passed to functions accepting `Number`s, and any operation performed is replaced with elementwise ONNX operators.

# Support
The focus of the package has been to export models from [Lux.jl](https://lux.csail.mit.edu/stable/), but much more work is needed to support all types of layers. See [Julia functions](docs/supported_functions.md), [ONNX operators](docs/supported_operators.md), and [Lux layers](docs/supported_layers.md) for details. There are also several limitations listed below, some of which might be solved in future versions.

## Limitations
ONNXExport cannot convert any arbitrary Julia function into ONNX, partly because of limitations in ONNXExport, but also due to limitations in ONNX itself.

### Dynamic dimensions
In principle, ONNXExport supports dynamic/symbolic dimensions, but this has not been fully tested and will likely fail for certain functions.

### Control flow
Due to the way ONNXExport traces Julia functions, it is not possible to capture certain control flow statements such as `if`, `for`, and `while`. These will likely result in the error `TypeError: non-boolean (ProbeNumber{Bool}) used in boolean context`. Try instead to use array operations or `ifelse`.

### Random
Random numbers from `rand` and `randn` will be treated as constants in the ONNX graph. The corresponding ONNX operators `RandomUniform` and `RandomNormal` are currently not supported.

### Broadcasting
Broadcasting support is limited to elementwise operations, e.g., `sin.(A)` and `relu.(Wx .+ b)`. Nested broadcasting or broadcasting over slices will likely fail.

### Immutability
ONNX tensors are immutable. Consequently, ONNXExport does not support mutating functions such as `setindex!`.

### Varargs functions
The tracing works by overloading commonly used functions with methods taking `ProbeArray` arguments instead of `AbstractArray`s. This poses problems for functions such as `cat`, which can accept an arbitrary number of arguments. The tracing might fail if none of the first few arguments are of type `ProbeArray`.

### Small integers and ONNX Runtime
The [ONNX documentation](https://onnx.ai/onnx/index.html) specifies that many operators, such as [Add](https://onnx.ai/onnx/operators/onnx__Add.html), support all real tensor data types, e.g., `int8` and `uint16`. However, this does not mean the various ONNX runtimes do (see [this issue](https://github.com/microsoft/onnxruntime/issues/19231)). `Float32`, `Int32`, and `Int64` are typically safe to use.

### Complex numbers
Complex numbers are currently not supported. Although the ONNX specification defines the `complex64` and `complex128` tensor data types, corresponding to the Julia types `ComplexF32` and `ComplexF64`, respectively, there are no operators that support them. The existing operators that use complex numbers, e.g., [DFT](https://onnx.ai/onnx/operators/onnx__DFT.html) and [ComplexMul](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftcomplexmul), use an interleaved representation, where the fastest changing dimension have size 2, corresponding to the real and imaginary parts. Conversion between complex Julia types and this array representation can be done with [reinterpret](https://docs.julialang.org/en/v1/base/arrays/#Base.reinterpret):
```julia
julia> size(reinterpret(reshape, Float32, rand(ComplexF32, 3, 4)))
(2, 3, 4)
```

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
