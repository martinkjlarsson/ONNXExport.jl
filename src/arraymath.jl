Base.:-(A::ProbeArray) = onnx_op("Neg", A)

for f in (:-, :*)
    @eval begin
        Base.$f(A::ProbeArray, B::AbstractArray) = $f(promote(A, B)...)
        Base.$f(A::AbstractArray, B::ProbeArray) = $f(promote(A, B)...)
        Base.$f(A::ProbeArray, B::ProbeArray) = $f(promote(A, B)...)
    end
end

@overload_splat Base.:+ _add ProbeArray AbstractArray 4
_add(A...) = _add(promote(A...)...)
_add(A::ProbeArray{T}) where {T} = A
_add(A::ProbeArray{T}, B::ProbeArray{T}) where {T} = onnx_op("Add", A, B)
function _add(
    A::ProbeArray{T}, B::ProbeArray{T}, C::ProbeArray{T}, D::ProbeArray{T}...
) where {T}
    return _add(A, _add(B, C, D...))
end
function _add(
    A::ProbeArray{T}, B::ProbeArray{T}, C::ProbeArray{T}, D::ProbeArray{T}...
) where {T<:AbstractFloat}
    return onnx_op("Sum", A, B, C, D...)
end

function Base.:-(A::ProbeArray{T}, B::ProbeArray{T}) where {T}
    return onnx_op("Sub", A, B)
end

# Multiplication with a scalar.
Base.:*(A::ProbeArray, B::Number) = A .* B
Base.:*(A::Number, B::ProbeArray) = A .* B

Base.:*(A::AbstractArray, B::ProbeNumber) = A .* B
Base.:*(A::ProbeNumber, B::AbstractArray) = A .* B

Base.:*(A::ProbeArray, B::ProbeNumber) = A .* B
Base.:*(A::ProbeNumber, B::ProbeArray) = A .* B

# Matrix-matrix and matrix-vector.
Base.:*(A::AbstractMatrix{T}, B::ProbeMatrix{T}) where {T} = *(probe(A), B)
Base.:*(A::ProbeMatrix{T}, B::AbstractMatrix{T}) where {T} = *(A, probe(B))

Base.:*(A::ProbeMatrix{T}, B::ProbeMatrix{T}) where {T} = matmul_onnx(A, B)
Base.:*(A::ProbeMatrix{T}, B::ProbeVector{T}) where {T} = matmul_onnx(A, B)

"""
Perform batched matrix multiplication.

The behavior depends on the arguments in the following way.
* If both arguments are 2-D they are multiplied like conventional matrices.
* If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the
  last two indexes and broadcast accordingly.
* If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its
  dimensions. After matrix multiplication the prepended 1 is removed.
* If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its
  dimensions. After matrix multiplication the appended 1 is removed.

See https://numpy.org/doc/stable/reference/generated/numpy.matmul.html.
"""
function matmul_onnx(A::ProbeArray{T}, B::ProbeArray{T}) where {T}
    ctx = GRAPH_CONTEXT[]

    op_type = "MatMul"
    nn = node_name(op_type)

    output_size = matmul_shape(raw_size(A), raw_size(B))
    output = value_info(eltype(A), output_size, output_name(nn))

    n = NodeProto(op_type, [name(B), name(A)], [name(output)]; name=nn)
    push!(ctx.nodes, n)

    return output_size == () ? ProbeNumber(output) : output
end
matmul_onnx(A::AbstractArray, B::ProbeArray) = matmul_onnx(promote(A, B)...)
matmul_onnx(A::ProbeArray, B::AbstractArray) = matmul_onnx(promote(A, B)...)
function matmul_onnx(A::ProbeArray, B::ProbeArray)
    @assert eltype(A) != eltype(B) "Wrong method was dispatched"
    return matmul_onnx(promote(A, B)...)
end
function matmul_onnx(A::AbstractArray{T}, B::AbstractArray{T}) where {T}
    A1d = length(A) == 1
    B1d = length(B) == 1

    if A1d
        A = reshape(A, 1, :)
    end
    if B1d
        B = reshape(B, :, 1)
    end

    @assert size(A, 2) == size(B, 1) "Mismatched dimensions"

    nd = max(ndims(A), ndims(B))
    dimsA = ntuple(i -> size(A, i), nd)
    dimsB = ntuple(i -> size(B, i), nd)

    A = reshape(A, dimsA)
    B = reshape(B, dimsB)

    repA = ones(Int, nd)
    repB = ones(Int, nd)
    for i in 1:nd
        if i > 2
            if dimsA[i] == 1
                repA[i] = dimsB[i]
            elseif dimsB[i] == 1
                repB[i] = dimsA[i]
            elseif dimsA[i] != dimsB[i]
                error("Mismatched dimensions")
            end
        end
    end

    A = repeat(A; inner=repA)
    B = repeat(B; inner=repB)

    lastdims = size(A)[3:end]

    A = reshape(A, size(A, 1), size(A, 2), :)
    B = reshape(B, size(B, 1), size(B, 2), :)

    C = similar(A, size(A, 1), size(B, 2), size(A, 3))
    for i in axes(C, 3)
        C[:, :, i] .= view(A, :, :, i) * view(B, :, :, i)
    end

    if A1d && B1d
        dimsC = lastdims
    elseif A1d
        dimsC = (size(B, 2), lastdims...)
    elseif B1d
        dimsC = (size(A, 1), lastdims...)
    else
        dimsC = (size(A, 1), size(B, 2), lastdims...)
    end
    C = reshape(C, dimsC)

    return C
end

# Uses the logic from NumPy (https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
# TODO: Make less ugly.
function matmul_shape(a, b)
    a1d = length(a) == 1
    b1d = length(b) == 1

    if a1d
        a = (1, a[1]) # Promote a to row matrix.
    end
    if b1d
        b = (b[1], 1) # Promote b to column matrix.
    end

    if a[2] != b[1]
        # TODO: Throw error if both are ints.
        @warn "Mismatched dimensions in MatMul; got $(a[2]) and $(b[1])"
    end

    c = ntuple(max(length(a), length(b))) do i
        i == 1 && return a[1]
        i == 2 && return b[2]
        return merge_dim(i <= length(a) ? a[i] : 1, i <= length(b) ? b[i] : 1)
    end

    if a1d && b1d
        c = c[3:end]
    elseif a1d
        c = c[2:end]
    elseif b1d
        c = (c[1], c[3:end]...)
    end

    return c
end

function Base.inv(A::ProbeMatrix)
    return onnx_op("Inverse", A; domain="com.microsoft")
end

function gemm(A::AbstractMatrix, B::AbstractMatrix; kwargs...)
    return gemm(promote(A, B)...; kwargs)
end
function gemm(A::AbstractMatrix, B::AbstractMatrix, C::AbstractArray; kwargs...)
    return gemm(promote(A, B, C)...; kwargs)
end
function gemm(
    A::ProbeMatrix{T},
    B::ProbeMatrix{T},
    C::Union{ProbeArray{T},Nothing}=nothing;
    alpha=one(T),
    beta=one(T),
    transA=false,
    transB=false,
) where {T}
    ctx = GRAPH_CONTEXT[]

    op_type = "Gemm"
    nn = node_name(op_type)

    new_dims = (raw_size(A, 1 + transA), raw_size(B, 2 - transB))
    # Switch A and B due to row-major order.
    inputs = isnothing(C) ? [name(B), name(A)] : [name(B), name(A), name(C)]
    output = value_info(T, new_dims, output_name(op_type))
    attr = (
        alpha=Float32(alpha), beta=Float32(beta), transA=Int(transA), transB=Int(transB)
    )

    n = NodeProto(op_type, inputs, [name(output)], attr; name=nn)
    push!(ctx.nodes, n)

    return output
end
