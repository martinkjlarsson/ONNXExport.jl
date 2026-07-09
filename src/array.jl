# TODO: Ensure all these support dynamically sizes arrays.

@overload_splat Base.fill _fill ProbeNumber{<:Integer} Integer 4 v::Number
@overload_splat Base.fill _fill ProbeNumber{<:Integer} Integer 4 v::ProbeNumber
_fill(v, dims...) = fill(v, dims)
Base.fill(v::ProbeNumber, dims::Integer...) = fill(v, dims)
function Base.fill(v::ProbeNumber, dims::NTuple{N,<:Integer}) where {N}
    shape = probe(collect(reverse(dims)), "shape")
    return onnx_op("Expand", dims, v, shape)
end
function Base.fill(v::ProbeNumber, dims::AtLeastOne{ProbeNumber{<:Integer},Integer})
    new_dims = map(d -> d isa Integer ? Int64(d) : dimension_name(), dims)
    shape = vcat(reverse(dims)...)
    return onnx_op("Expand", new_dims, v, shape)
end
function Base.fill(v::Number, dims::AtLeastOne{ProbeNumber{<:Integer},Integer})
    check_probe(v)
    new_dims = map(d -> d isa Integer ? Int64(d) : dimension_name(), dims)
    shape = vcat(reverse(dims)...)
    return onnx_op(
        "ConstantOfShape", typeof(v), new_dims, shape; attr=(value=TensorProto([v]),)
    )
end
Base.fill(v::ProbeNumber, ::Tuple{}) = ProbeArray(v)

# TODO: Should we mark the ProbeArray as uninitialized somehow?
function Base.similar(
    A::ProbeArray, element_type::Type, dims::NTuple{N,Int} where {N}=size(A)
)
    return ProbeArray{element_type}("uninitialized", dims)
end

function Base.reshape(
    A::ProbeArray, dims::NTuple{N,Union{Colon,Integer,ProbeNumber{<:Integer}}}
) where {N}
    return _reshape(A, dims)
end
function Base.reshape(A::ProbeArray, dims::NTuple{N,Union{Colon,Int64}}) where {N}
    return _reshape(A, dims)
end
Base.reshape(A::ProbeArray, dims::NTuple{N,Int64}) where {N} = _reshape(A, dims)
function Base.reshape(
    A::ProbeArray, dims::Vararg{Union{Colon,Integer,ProbeNumber{<:Integer}},N}
) where {N}
    return _reshape(A, dims)
end
function Base.reshape(A::ProbeArray, dims::Vararg{Union{Colon,Int64},N}) where {N}
    return _reshape(A, dims)
end
Base.reshape(A::ProbeArray, dims::Vararg{Int64,N}) where {N} = _reshape(A, dims)

# TODO: Detect no-op reshapes, e.g., reshape(A::Vector, sz)?
function _reshape(A::ProbeArray, ::Tuple{})
    return onnx_op("Reshape", (), A, probe(Int64[], "shape"))
end
function _reshape(
    A::ProbeArray, dims::NTuple{N,Union{Colon,Integer,ProbeNumber{<:Integer}}}
) where {N}
    reshape_dims = Vector{Union{Int64,ProbeNumber{Int64}}}(undef, length(dims))
    new_dims = Vector{ProbeDim}(undef, length(dims))

    # Find the symbolic indices and the product of the constants.
    sym_dims = Int[]
    const_dims = 1
    for i in eachindex(dims)
        if dims[i] === (:) || isprobe(dims[i])
            push!(sym_dims, i)
        else
            d = Int64(dims[i])
            const_dims *= d

            reshape_dims[i] = d
            new_dims[i] = d
        end
    end

    sym_A = Int[]
    const_A = 1
    for i in 1:ndims(A)
        if raw_size(A, i) isa Symbol
            push!(sym_A, i)
        else
            const_A *= raw_size(A, i)
        end
    end

    if isempty(sym_dims)
        # new_dims and reshape_dims are already fully assigned.
        # NOTE: If A has a single symbolic dimension, we can infer it here.
    elseif isempty(sym_A) && length(sym_dims) == 1
        # We can infer the symbolic dimension in dims from A.
        i = only(sym_dims)
        newd = length(A) ÷ const_dims
        new_dims[i] = newd
        reshape_dims[i] = newd
    elseif length(sym_A) == 1 && length(sym_dims) == 1 && const_dims == const_A
        # If the constant parts are identical, we can keep the symbol from A. This
        # introduces fewer dimension names and thus fewer "dimension mismatched" warnings.
        i = only(sym_dims)
        new_dims[i] = raw_size(A, only(sym_A))
        reshape_dims[i] = -1
    elseif length(sym_dims) == 1
        # We can atleast utilize the single -1/Colon-dimension.
        i = only(sym_dims)
        new_dims[i] = dimension_name()
        reshape_dims[i] = -1
    else
        # Nothing clever to do.
        for i in sym_dims
            if dims[i] === (:)
                new_dims[i] = dimension_name()
                reshape_dims[i] = -1
            elseif isprobe(dims[i])
                new_dims[i] = dimension_name()
                reshape_dims[i] = convert(ProbeNumber{Int64}, dims[i])
            else
                error("This should not happen")
            end
        end
    end

    shape = probe(vcat(reverse(reshape_dims)...))
    return onnx_op("Reshape", Tuple(new_dims), A, shape)
end

Base.vec(x::ProbeScalar) = reshape(x, 1)
Base.vec(v::ProbeVector) = v
Base.vec(A::ProbeArray) = reshape(A, :)

# Transpose does not have to be lazy. Unnecessary ONNX operators can be optimized away in
# post-processing.
Base.transpose(x::ProbeNumber) = x
Base.transpose(x::ProbeScalar) = x
Base.transpose(v::ProbeVector) = reshape(v, 1, :)
Base.transpose(A::ProbeMatrix) = onnx_op("Transpose", reverse(raw_size(A)), A)

Base.adjoint(A::ProbeArray) = transpose(A)
Base.adjoint(::ProbeArray{<:Complex}) = error("ONNX does not support complex conjugation.")

Base.permutedims(v::ProbeVector) = reshape(v, 1, :)
Base.permutedims(A::ProbeMatrix) = onnx_op("Transpose", reverse(raw_size(A)), A)
function Base.permutedims(A::ProbeArray, perm::NTuple{N,Int}) where {N}
    out_dims = ntuple(i -> raw_size(A, perm[i]), N)
    onnx_perm = reverse(N .- perm)
    return onnx_op("Transpose", out_dims, A; attr=(perm=onnx_perm,))
end

Base.PermutedDimsArray(A::ProbeArray, perm::NTuple{N,Int}) where {N} = permutedims(A, perm)

# TODO: Consider implementing repeat(A::AbstractArray, counts::ProbeDim...) which can be dispatched.
Base.repeat(A::ProbeArray; inner=nothing, outer=nothing) = _repeat(A, inner, outer)
_repeat(A::ProbeArray, ::Nothing, ::Nothing) = A
function _repeat(A::ProbeArray, inner, ::Nothing)
    n = max(ndims(A), length(inner))
    return repeat_inner_outer(A, pad(inner, n), pad(nothing, n))
end
function _repeat(A::ProbeArray, ::Nothing, outer)
    n = max(ndims(A), length(outer))
    return repeat_outer(A, pad(outer, n))
end
function _repeat(A::ProbeArray, inner, outer)
    n = max(ndims(A), length(inner), length(outer))
    return repeat_inner_outer(A, pad(inner, n), pad(outer, n))
end

pad(::Nothing, n) = ntuple(Returns(1), n)
pad(i::Int, n) = (i, ntuple(Returns(1), n - 1)...)
pad(dims::Dims{N}, n) where {N} = (dims..., ntuple(Returns(1), n - N)...)
pad(itr, n) = (itr..., ntuple(Returns(1), n - length(itr))...)

function interleave(a::NTuple{N}, b::NTuple{N}) where {N}
    return ntuple(i -> isodd(i) ? a[(i + 1) ÷ 2] : b[i ÷ 2], 2N)
end

const IntsOrProbeInts{N} = NTuple{N,Union{Int,ProbeNumber{<:Integer}}}
function repeat_outer(A::ProbeArray, dims::IntsOrProbeInts{N}) where {N}
    # Add new dimensions if necessary.
    if ndims(A) < N
        A = unsqueeze(A, (ndims(A) + 1):N)
    end

    repeats = probe(vcat(reverse(dims)...))
    new_dims = mul_dim.(raw_size(A), raw_dims(dims))
    return onnx_op("Tile", new_dims, A, repeats)
end

function repeat_inner_outer(
    A::ProbeArray, inner::IntsOrProbeInts{N}, outer::IntsOrProbeInts{N}
) where {N}
    B = unsqueeze(A, 1:2:(2 * ndims(A)))
    B = repeat_outer(B, interleave(inner, outer))
    B = reshape(B, size(A) .* (inner .* outer))
    return B
end

# TODO: Replace ProbeArray with Union{ProbeArray, ProbeNumber} or similar.
@overload_splat Base.vcat _vcat_forward ProbeNumber Number 4 # TODO: Temp fix.

@overload_splat Base.vcat _vcat_forward ProbeArray Union{AbstractArray,Number} 4
@overload_splat Base.hcat _hcat_forward ProbeArray Union{AbstractArray,Number} 4
@overload_splat Base.cat _cat_forward ProbeArray Union{AbstractArray,Number} 4

# We need these because of the methods in SparseArrays.jl.
@overload_splat Base.vcat _vcat_forward ProbeArray Union{AbstractVecOrMat{<:Number},Number} 4
@overload_splat Base.hcat _hcat_forward ProbeArray Union{AbstractVecOrMat{<:Number},Number} 4

@overload_splat Base.hvcat _hvcat ProbeArray Union{AbstractArray,Number} 4 rows::Tuple{
    Vararg{Int}
}
@overload_splat Base.hvcat _hvcat ProbeArray Union{AbstractVecOrMat{<:Number},Number} 4 rows::Tuple{
    Vararg{Int}
}

_vcat_forward(A...) = _cat(1, A...)
_hcat_forward(A...) = _cat(2, A...)
_cat_forward(A...; dims) = _cat(dims, A...)
_cat(::Val{dims}, A...) where {dims} = _cat(dims, A...)
_cat(dims, As...) = _cat(dims, promote(As...)...)
function _cat(::Dims, ::ProbeArray{T}...) where {T}
    return error("ONNX export does not support concatination over multiple dimensions")
end
function _cat(dims::Int, As::Union{ProbeArray{T},ProbeNumber{T}}...) where {T}
    # Ensure all tensors have the same dimensionality and include dims.
    N = max(maximum(ndims.(As)), dims)
    As = map(A -> ndims(A) == N ? A : unsqueeze(A, (ndims(A) + 1):N), As)

    sum_dims = Tuple(raw_size(A, dims) for A in As)
    total = sum_dims isa Dims ? sum(sum_dims) : dimension_name()
    new_dims = ntuple(N) do i
        return i == dims ? total : reduce(_cat_dims, raw_size(A, i) for A in As)
    end

    return onnx_op("Concat", new_dims, As...; attr=(axis=N - dims,))
end
function _cat_dims(a::Int, b::Int)
    return a == b ? a : throw(DimensionMismatch("got dimension with sizes $a and $b"))
end
_cat_dims(a::Int, b::Symbol) = _cat_dims(b, a)
function _cat_dims(a::Symbol, b::Int)
    @warn "Dimensions do not match, assuming dynamic dimension $a=$b"
    return b
end
function _cat_dims(a::Symbol, b::Symbol)
    if a != b
        @warn "Dimensions do not match, assuming dynamic dimensions $a and $b are equal"
    end
    return a
end

_hvcat(rows, A...) = _hvcat(rows, promote(A...)...)
function _hvcat(rows, A::ProbeArray{T}...) where {T}
    @assert sum(rows) == length(A)

    starts = cumsum(rows) .- rows .+ 1
    ends = starts .+ rows .- 1

    return vcat([hcat(A[starts[i]:ends[i]]...) for i in eachindex(starts)]...)
end

function Base.accumulate(
    op,
    A::ProbeArray{T};
    dims::Union{Integer,Nothing}=nothing,
    init::Union{Number,Nothing}=nothing,
) where {T}
    if isnothing(dims)
        # TODO: This results in unnecessary rehapes for vectors.
        #       This is not a huge issue but perhaps avoidable.
        return reshape(accumulate(op, vec(A); dims=1, init), size(A))
    end
    if dims > ndims(A)
        return A
    end

    if isnothing(init)
        init = init_elem(eltype(A), op)
    end

    if op == (+) && iszero(init)
        axis = probe(ndims(A) - dims, "axis")
        return onnx_op("CumSum", A, axis)
    end
    # TODO: CumProd requires opset 26. Add export options to choose implementation.
    # if op == (*) && isone(init)
    #     axis = probe(ndims(A) - dims, "axis")
    #     return onnx_op("CumProd", A, axis)
    # end

    slice_size = _selectdimsize(raw_size(A), dims)
    init_tensor = probe(fill(init, slice_size))

    # TODO: To handle init properly, we can add one more input to scan, first=true, which
    #       is only true the first iteration and bypasses the op-operator. This might be
    #       slow if and If operator is needed.
    _, scan_output = scan_onnx(
        (init_tensor,), (A,); scan_input_axes=(dims,), scan_output_axes=(dims,)
    ) do state_in, scan_in
        out = op.(state_in, scan_in)
        return out, out
    end

    return scan_output[1]
end
init_elem(::Type{T}, ::typeof(+)) where {T} = zero(T)
init_elem(::Type{T}, ::typeof(*)) where {T} = one(T)
init_elem(::Type{T}, ::typeof(max)) where {T} = typemin(T)
init_elem(::Type{T}, ::typeof(min)) where {T} = typemax(T)
function init_elem(::Type{T}, f) where {T}
    return error(
        "ONNX export of accumulate with element type $T and op $f without init is " *
        "currently not supported. Provide init or file an issue.",
    )
end

Base.cumprod(A::ProbeVector) = cumprod(A; dims=1)
Base.cumprod(A::ProbeArray; dims::Integer) = accumulate(*, conv_cum(A); dims=dims)

Base.cumsum(A::ProbeVector) = cumsum(A; dims=1)
Base.cumsum(A::ProbeArray; dims::Integer) = accumulate(+, conv_cum(A); dims=dims)

function conv_cum(A)
    et = eltype(A)
    if et <: Signed && promote_type(et, Int) == Int
        A = convert(ProbeArray{Int}, A)
    elseif et <: Unsigned && promote_type(et, UInt) == UInt
        A = convert(ProbeArray{UInt}, A)
    end
    return A
end

Base.dropdims(A::ProbeArray; dims) = _dropdims(A, dims)
_dropdims(A::ProbeArray, dims::Integer) = dropdims(A; dims=(Int(dims),))
function _dropdims(A::ProbeArray, dims::Dims{N}) where {N}
    new_size = foldl(
        (ds, d) -> d ∈ dims ? ds : (ds..., raw_size(A, d)), 1:ndims(A); init=()
    )
    axes = probe(collect(ndims(A) .- dims))
    return onnx_op("Squeeze", new_size, A, axes)
end

unsqueeze(x::Number, dims) = unsqueeze(fill(x), dims)
unsqueeze(A::AbstractArray, dims::Int) = unsqueeze(A, (dims,))
unsqueeze(A::AbstractArray, dims::AbstractVector{Int}) = unsqueeze(A, Tuple(dims))
function unsqueeze(A::AbstractArray, dims::Dims{N}) where {N}
    M = ndims(A) + N
    if any(∉(1:M), dims)
        throw(ArgumentError(lazy"the new dimensions must be in the range 1:$M; got $dims"))
    end
    if !allunique(dims)
        throw(ArgumentError(lazy"new dimensions have repeated entries; got $dims"))
    end
    new_dims = unsqueeze_size(size(A), dims)
    return reshape(A, new_dims)
end
function unsqueeze(A::ProbeArray, dims::Dims{N}) where {N}
    N == 0 && return A

    M = ndims(A) + N
    new_dims = unsqueeze_size(raw_size(A), dims)
    axes = probe(collect(M .- dims), "axes")
    return onnx_op("Unsqueeze", new_dims, A, axes)
end
function unsqueeze_size(s, dims)
    return ntuple(length(s) + length(dims)) do i
        lt = 0
        for d in dims
            d == i && return 1
            d <= i && (lt += 1)
        end
        return s[i - lt]
    end
end

function Base.partialsort(
    v::ProbeVector, k::Union{Integer,OrdinalRange}; by=identity, lt=isless, rev=false
)
    @assert lt === isless "ONNX export only supports lt=isless"

    maxk = maximum(k)
    new_dims = (maxk,)
    kk = probe([maxk], "k")

    values = value_info(eltype(v), new_dims, "values")
    idxs = value_info(Int64, new_dims, "indices")
    onnx_op("TopK", (by.(v), kk), (values, idxs); attr=(largest=rev,))
    if by !== identity
        values = v[idxs]
    end

    k == 1:maxk && return values
    k == 1 && return reshape(values)
    return values[k]
end
