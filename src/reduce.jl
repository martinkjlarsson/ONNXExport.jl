Base.unique(A::ProbeArray; dims=:) = _unique(A, dims)
Base.unique(f, A::ProbeArray) = _unique(f, A)

Base.maximum(A::ProbeArray; dims=:) = _reduce("ReduceMax", A, dims)
Base.maximum(f, A::ProbeArray; dims=:) = maximum(f.(A); dims=dims)

Base.minimum(A::ProbeArray; dims=:) = _reduce("ReduceMin", A, dims)
Base.minimum(f, A::ProbeArray; dims=:) = minimum(f.(A); dims=dims)

# Base.extrema returns an array of tuples, which does not map nicely to ONNX.
function Base.extrema(A::ProbeArray; dims=:)
    dims == (:) || error("dims argument is not supported for ONNX export")
    return (minimum(A), maximum(A))
end
function Base.extrema(f, A::ProbeArray; dims=:)
    dims == (:) || error("dims argument is not supported for ONNX export")
    return (minimum(f, A), maximum(f, A))
end

# Base.argmax and Base.argmin return CartesianIndex which does not map nicely to ONNX.
function Base.argmax(v::ProbeVector; dims=:)
    dims == (:) || error("dims argument is not supported for ONNX export")
    return _argminmax("ArgMax", v)
end
function Base.argmax(f, v::ProbeVector; dims=:)
    return error("f argument is not supported for ONNX export")
end
function Base.argmin(v::ProbeVector; dims=:)
    dims == (:) || error("dims argument is not supported for ONNX export")
    return _argminmax("ArgMin", v)
end
function Base.argmin(f, v::ProbeVector; dims=:)
    return error("f argument is not supported for ONNX export")
end

Base.sum(A::ProbeArray; dims=:) = _reduce("ReduceSum", widen_ints_sum(A), dims)
Base.sum(f, A::ProbeArray; dims=:) = sum(f.(A); dims=dims)
function Base.sum(::typeof(abs), A::ProbeArray; dims=:)
    return _reduce("ReduceL1", widen_ints_sum(A), dims)
end
function Base.sum(::typeof(abs2), A::ProbeArray{<:Real}; dims=:)
    return _reduce("ReduceSumSquare", widen_ints_sum(A), dims)
end

Base.prod(A::ProbeArray; dims=:) = _reduce("ReduceProd", widen_ints_prod(A), dims)
Base.prod(f, A::ProbeArray; dims=:) = prod(f.(A); dims=dims)

Base.any(A::ProbeArray{Bool}; dims=:) = _reduce("ReduceMax", A, dims)
Base.any(p::Function, A::ProbeArray; dims=:) = any(p.(A); dims=dims)

Base.all(A::ProbeArray{Bool}; dims=:) = _reduce("ReduceMin", A, dims)
Base.all(p::Function, A::ProbeArray; dims=:) = all(p.(A); dims=dims)

function Base.count(A::ProbeArray{Bool}; dims=:)
    A = convert(ProbeArray{Int}, A)
    return sum(A; dims=dims)
end
function Base.count(f, A::ProbeArray; dims=:)
    return count(f.(A); dims=dims)
end

# Base.reduce delegates to mapreduce with f=identity.

# We can splat the vector as performance is not a concern here. If the Julia compilation is
# terribly slow the ONNX model will likely be as well.
Base.reduce(::typeof(vcat), A::AbstractVector{<:ProbeArray}) = _cat(1, A...)
Base.reduce(::typeof(hcat), A::AbstractVector{<:ProbeArray}) = _cat(2, A...)

# TODO: Should we accept the init keyword argument but verify it is neutral?
function Base.mapreduce(f, op, ::ProbeArray; kwargs...)
    return error("ONNX export of mapreduce with f=$f and op=$op is not supported.")
end
Base.mapreduce(f, ::typeof(+), A::ProbeArray; dims=:) = _reduce("ReduceSum", f.(A), dims)
function Base.mapreduce(::typeof(abs), ::typeof(+), A::ProbeArray; dims=:)
    return _reduce("ReduceL1", A, dims)
end
function Base.mapreduce(::typeof(abs2), ::typeof(+), A::ProbeArray{<:Real}; dims=:)
    return _reduce("ReduceSumSquare", A, dims)
end
function Base.mapreduce(f, ::typeof(*), A::ProbeArray; dims=:)
    return _reduce("ReduceProd", f.(A), dims)
end
Base.mapreduce(f, ::typeof(min), A::ProbeArray; dims=:) = _reduce("ReduceMin", f.(A), dims)
Base.mapreduce(f, ::typeof(max), A::ProbeArray; dims=:) = _reduce("ReduceMax", f.(A), dims)
function Base.mapreduce(f, ::typeof(|), A::ProbeArray{Bool}; dims=:)
    return _reduce("ReduceMax", f.(A), dims)
end
function Base.mapreduce(f, ::typeof(&), A::ProbeArray{Bool}; dims=:)
    return _reduce("ReduceMin", f.(A), dims)
end
function Base.mapreduce(
    f, ::typeof(vcat), A::AbstractVector{<:Union{ProbeArray,ProbeNumber}}
)
    return _cat(1, f.(A)...)
end
function Base.mapreduce(
    f, ::typeof(hcat), A::AbstractVector{<:Union{ProbeArray,ProbeNumber}}
)
    return _cat(2, f.(A)...)
end

_reduce(op_type::String, A::ProbeArray, dims::Integer) = _reduce(op_type, A, (dims,))
function _reduce(op_type::String, A::ProbeArray, ::Colon)
    return onnx_op(op_type, (), A; attr=(keepdims=0,))
end
function _reduce(op_type::String, A::ProbeArray, dims)
    new_dims = ntuple(i -> i ∈ dims ? 1 : raw_size(A, i), ndims(A))
    axes = probe(collect(ndims(A) .- dims), "axes")
    return onnx_op(op_type, new_dims, A, axes)
end

function _argminmax(op_type::String, v::ProbeVector)
    return onnx_op(op_type, Int64, (), v; attr=(keepdims=0,)) + one(Int64)
end

function _unique(A::ProbeArray, ::Colon)
    new_dims = dimension_name()
    return onnx_op("Unique", (new_dims,), A; attr=(sorted=0,))
end
function _unique(A::ProbeArray, dims::Int)
    1 <= dims <= ndims(A) || return A
    raw_size(A, dims) == 1 && return A

    new_dim = dimension_name()
    new_dims = ntuple(i -> i == dims ? new_dim : raw_size(A, i), ndims(A))
    return onnx_op("Unique", new_dims, A; attr=(axis=ndims(A) - dims, sorted=0))
end
function _unique(f, A::ProbeArray)
    new_dims = dimension_name()
    values = value_info(eltype(A), (new_dims,), "unique_values")
    indices = value_info(Int64, (new_dims,), "unique_indices")
    onnx_op("Unique", (f.(A),), (values, indices); attr=(sorted=0,))

    return A[indices .+ 1]
end
