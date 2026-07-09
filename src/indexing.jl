# We want to overload Base.getindex such that our implementation is called if any of the
# indices are of the ProbeArray type. Julia has no elegant way of doing this. Tne approach
# here is to explictely declare every possible combination of ProbeArray and Any in the
# index arguments up to some arity. The is done using the probe_getindex macro.

macro probe_getindex(first_type, N)
    N = eval(N)
    methods = Expr[]

    for mask in 1:(2^N - 1)
        n = 64 - leading_zeros(Int64(mask))

        arglist = []
        callargs = []

        for i in 1:n
            if ((mask >> (i - 1)) & 1) == 1
                push!(arglist, :($(Symbol(:I, i))::ProbeArray))
            else
                push!(arglist, Symbol(:I, i))
            end
            push!(callargs, Symbol(:I, i))
        end

        push!(arglist, :(Irest...))
        push!(callargs, :(Irest...))

        push!(
            methods,
            quote
                Base.getindex(A::$first_type, $(arglist...)) = _getindex(A, $(callargs...))
            end,
        )
    end

    return Expr(:block, methods...)
end

Base.IndexStyle(::ProbeArray) = IndexLinear()

# Create methods up to arity 4. This means that the array A and the 3 first indices can be
# non-ProbeArray constants, and we will still detect the forth being a ProbeArray.
Base.getindex(A::ProbeArray, Irest...) = _getindex(A, Irest...)
@probe_getindex ProbeArray 4
@probe_getindex AbstractArray 4

Base.getindex(x::ProbeNumber) = x
Base.getindex(A::ProbeScalar) = ProbeNumber(A)

# Indexing with a single bool array.
_getindex(A::ProbeArray{T,N}, I::AbstractArray{Bool,N}) where {T,N} = compress(A, vec(I))
_getindex(A::ProbeArray{T,N}, I::ProbeArray{Bool,N}) where {T,N} = compress(A, vec(I))
_getindex(A::ProbeArray, I::ProbeVector{Bool}) = compress(A, I)
_getindex(A::ProbeVector, I::ProbeVector{Bool}) = compress(A, I)
_getindex(A::ProbeVector, I::AbstractVector{Bool}) = compress(A, I)

function compress(A::ProbeArray, I::AbstractVector{Bool})
    condition = probe(I, "condition")
    return onnx_op("Compress", (count(I),), A, condition)
end
function compress(A::ProbeArray, I::ProbeVector{Bool})
    new_dim = dimension_name()
    return onnx_op("Compress", (new_dim,), A, I)
end

# General indexing.
_getindex(A, I) = _getindex(vec(A), I)
function _getindex(A::AbstractArray{T,N}, I::Vararg{Any,N}) where {T,N}
    A = probe(A, "array")
    return _getindex(A, I...)
end
function _getindex(A::ProbeArray{T,N}, I::Vararg{Any,N}) where {T,N}
    A, I = try_slicing(A, I)

    # Iterate backwards in case a Gather operator reduces the number of dimensions.
    for dim in length(I):-1:1
        A = index_dimension(A, dim, I[dim])
    end

    return A
end

index_dimension(A::ProbeArray, ::Int, ::Colon) = A
function index_dimension(A::ProbeArray{T,N}, dim::Int, I::AbstractVector{Bool}) where {T,N}
    condition = probe(I, "condition")
    dims = (raw_size(A)[1:(dim - 1)]..., count(I), raw_size(A)[(dim + 1):N]...)
    return onnx_op("Compress", dims, A, condition; attr=(axis=N - dim,))
end
function index_dimension(A::ProbeArray{T,N}, dim::Int, I::ProbeVector{Bool}) where {T,N}
    new_dim = dimension_name()
    dims = (raw_size(A)[1:(dim - 1)]..., new_dim, raw_size(A)[(dim + 1):N]...)
    return onnx_op("Compress", dims, A, I; attr=(axis=N - dim,))
end
function index_dimension(
    A::ProbeArray{T,N},
    dim::Int,
    I::Union{AbstractVector{<:Integer},AbstractArray{<:Integer,0}},
) where {T,N}
    I0 = I .- one(eltype(I)) # Convert to zero-based indices.
    I0 = probe(I0)

    if promote_type(eltype(I0), Int32) == Int32
        I0 = convert(ProbeArray{Int32}, I0)
    else
        I0 = convert(ProbeArray{Int64}, I0)
    end

    dims = (raw_size(A)[1:(dim - 1)]..., raw_size(I0)..., raw_size(A)[(dim + 1):N]...)
    return onnx_op("Gather", dims, A, I0; attr=(axis=N - dim,))
end
function index_dimension(
    A::ProbeArray{T,N}, dim::Int, I::Union{ProbeNumber{<:Integer},Integer}
) where {T,N}
    I0 = I - one(eltype(I)) # Convert to zero-based indices.
    I0 = probe(I0)

    if promote_type(eltype(I0), Int32) == Int32
        I0 = convert(ProbeNumber{Int32}, I0)
    else
        I0 = convert(ProbeNumber{Int64}, I0)
    end

    new_dims = (raw_size(A)[1:(dim - 1)]..., raw_size(A)[(dim + 1):N]...)
    return onnx_op("Gather", new_dims, A, I0; attr=(axis=N - dim,))
    # TODO: What if the result is a scalar
end

"""
    try_slicing(A::ProbeArray{T,N}, I::NTuple{Any,N}) where {T,N}

Try to convert all constant indices in `I` to ranges and use the Slice operator to index
over all those dimensions simultaneously. Returns an array `B` and indices `J` such that
`A[I] = B[J]`.
"""
function try_slicing(A::ProbeArray{T,N}, I::NTuple{N,Any}) where {T,N}
    int_mask = ntuple(i -> I[i] isa Integer, N)

    Ir = ntuple(i -> try_to_range(I[i]), N)
    A = slice(A, Ir...)
    I = ntuple(i -> Ir[i] == (:) ? I[i] : (:), N)

    # Drop singleton dimensions resulting from indexing with a constant scalar.
    int_idxs = findall(int_mask)
    if !isempty(int_idxs)
        keep_mask = collect(.!(int_mask))
        dims = raw_size(A)[keep_mask]
        I = I[keep_mask]

        axes = probe(Int64.(N .- int_idxs), "axes")
        A = onnx_op("Squeeze", dims, A, axes)
    end

    return A, I
end

function try_to_range(i)
    @warn "Indexing with type $(typeof(i)) might be convertible to a range. Add support for this."
    return (:)
end
try_to_range(::Colon) = (:)
try_to_range(::ProbeArray) = (:)
# While ProbeNumber could be used with Slice, it would require concatenation operations to
# produce the starts and ends tensors.
try_to_range(::ProbeNumber) = (:)
try_to_range(i::Integer) = i:i
try_to_range(i::OrdinalRange{<:Integer}) = i
try_to_range(i::AbstractVector) = isempty(i) ? (2:1) : (:)
try_to_range(i::AbstractVector{Bool}) = try_to_range(findall(i))
function try_to_range(i::AbstractVector{T}) where {T<:Integer}
    isempty(i) && return T(2):T(1) # Empty range.
    length(i) == 1 && return first(i):first(i)

    start = first(i)
    stop = last(i)
    step = i[begin + 1] - start

    iszero(step) && return (:)

    r = start:step:stop

    return r == i ? r : (:)
end

slice(A::ProbeArray{T,N}, ::Vararg{Colon,N}) where {T,N} = A
function slice(
    A::ProbeArray{T,N}, I::Vararg{Union{OrdinalRange{<:Integer},Colon},N}
) where {T,N}
    starts = Int[]
    ends = Int[]
    axes = Int[]
    steps = Int[]
    for i in N:-1:1
        if I[i] === (:) || (isa(raw_size(A, i), Int) && I[i] == 1:raw_size(A, i))
            continue
        end
        push!(starts, first(I[i]) - 1)
        # If we index backwards to the first element, we cannot set end to -1 as it
        # represents the last element.
        if last(I[i]) == 1 && step(I[i]) < 0
            push!(ends, typemin(Int))
        else
            push!(ends, last(I[i]) - 1 + sign(step(I[i])))
        end
        push!(axes, N - i)
        push!(steps, step(I[i]))
    end
    if isempty(starts)
        return A
    end

    dims = ntuple(i -> I[i] == (:) ? size(A, i) : length(I[i]), N)

    starts = probe(starts, "starts")
    ends = probe(ends, "ends")
    axes = probe(axes, "axes")
    steps = probe(steps, "steps")
    return onnx_op("Slice", dims, A, starts, ends, axes, steps)
end
