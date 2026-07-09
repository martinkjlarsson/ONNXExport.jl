# ONNXExport

[![Build Status](https://github.com/martinkjlarsson/ONNXExport.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/martinkjlarsson/ONNXExport.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia function `f(inputs...)` can be exported to ONNX using

```julia
export_model("model.onnx", f, inputs...)
```

# Design
ONNXExport works by defining a custom `AbstractArray` subtype `ProbeArray` and `Number` subtype `ProbeNumber`. These can be passed to many functions that normally accept `AbstractArray`s or `Number`s, e.g., arithmetic and linear algebra operations, neural networks, and array manipulations. Instead of performing the operations, the `Probe*` types write operators to an ONNX graph, which can later be saved to file. This enables the tracing of a Julia function with export to ONNX.

Broadcasting is also supported through the `BroadcastProbe <: Number` type. It wraps a `ProbeArray` such that the array can be passed to functions accepting `Number`s and any operation performed is replaced with elementwise ONNX operators.

# Indexing in Julia and ONNX
Julia uses [column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order) (like Fortran and MATLAB), while ONNX uses row-major order (like C and Python/NumPy). In Julia, the leftmost index varies fastest, while in ONNX, the rightmost index varies fastest. To solve this discrepancy, we reverse the dimensions when writing Julia arrays as ONNX tensors. A Julia array of size `(row, column, batch)` is written to ONNX as a tensor of shape `(batch, column, row)`. Note that the data remains unchanged, e.g., `row` is the fastest varying dimension in both cases.

However, ONNX expects tensors on the form `(batch, row, column)` for matrix operations, so we have effectively transposed the rows and columns. The solution to this is to transpose all matrix operations.
For example, a linear layer might perform the operation
```math
y = Wx + b.
```
When converted to ONNX, we are performing the operations
```math
y^T = x^T W^T + b^T,
```
where the transposes come for free due to the reversal of the dimensions. Importantly, we must switch the argument order in the matrix multiplication: $Wx \rightarrow x^T W^T$.

Furthermore, Julia uses one-based indexing, while ONNX uses zero-based indexing. The conversion is trivial in most cases, but care is required when indexing dimensions. Dimension `d` in Julia corresponds to dimension `ndims(A) - d` in ONNX. For example, `permutedims(A, (2, 1, 3, 4))` in Julia, i.e., permuting the two fastest varying/leftmost dimensions, will be written to ONNX as a [Transpose](https://onnx.ai/onnx/operators/onnx__Transpose.html) operator with the `perm` attribute `(0, 1, 3, 2)`, i.e., permuting the two fastest varying/rightmost dimensions.
