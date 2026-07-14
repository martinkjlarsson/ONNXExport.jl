const ProbeDim = Union{Int,Symbol}
const ProbeDims{N} = NTuple{N,ProbeDim}

mul_dim(a::Int, b::Int) = a * b
mul_dim(::ProbeDim, ::ProbeDim) = dimension_name()
div_dim(a::Int, b::Int) = a ÷ b
div_dim(::ProbeDim, ::ProbeDim) = dimension_name()
add_dim(a::Int, b::Int) = a + b
add_dim(::ProbeDim, ::ProbeDim) = dimension_name()

struct ProbeArray{T,N} <: AbstractArray{T,N}
    name::String
    size::ProbeDims{N}
    # TODO: Add tuple for denotations?

    function ProbeArray{T,N}(name::String, size::ProbeDims{N}) where {T,N}
        check_probe(T)
        return new{T,N}(name, size)
    end
end

const ProbeMatrix{T} = ProbeArray{T,2}
const ProbeVector{T} = ProbeArray{T,1}
const ProbeScalar{T} = ProbeArray{T,0}
const ProbeVecOrMat{T} = Union{ProbeVector{T},ProbeMatrix{T}}
const AbstractScalar = Union{Number,AbstractString}

const ExactlyOne{S,T} = Union{
    Tuple{S,Vararg{T}},
    Tuple{T,S,Vararg{T}},
    Tuple{T,T,S,Vararg{T}},
    Tuple{T,T,T,S,Vararg{T}},
    # Tuple{T,T,T,T,S,Vararg{T}},
    # Tuple{T,T,T,T,T,S,Vararg{T}},
}
const AtLeastOne{S,T} = ExactlyOne{S,Union{S,T}}
"A Tuple with at least one element being a ProbeArray."
const ProbeTuple = ExactlyOne{ProbeArray,Any}

function ProbeArray{T}(name::String, dims::Vararg{ProbeDim,N}) where {T,N}
    return ProbeArray{T,N}(name, dims)
end
function ProbeArray{T}(name::String, dims::ProbeDims{N}) where {T,N}
    return ProbeArray{T,N}(name, dims)
end
ProbeArray{T}(A::ProbeArray) where {T} = ProbeArray{T,ndims(A)}(A::ProbeArray)
function ProbeArray{T,N}(A::ProbeArray{S,N}) where {T,N,S}
    if S == T
        @warn "Converting from ProbeArray{$S} to ProbeArray{$T} is a no-op. This should not happen."
        # Perform cast anyway so we create a new ProbeArray instance.
    end
    # TODO: Is this hacky?
    if N == 0
        return ProbeArray(onnx_op("Cast", T, A; attr=(to=Int(tensor_type(T)),)))
    else
        return onnx_op("Cast", T, A; attr=(to=Int(tensor_type(T)),))
    end
end

Base.convert(::Type{T}, A::ProbeArray) where {T<:ProbeArray} = A isa T ? A : T(A)::T
function Base.convert(::Type{T}, A::AbstractArray) where {T<:ProbeArray}
    # Convert to ProbeArray first, then convert the element type.
    return convert(T, probe(A))::ProbeArray
end
function Base.convert(::Type{ProbeArray{T}}, x::AbstractScalar) where {T<:AbstractScalar}
    # Convert scalar first, then convert to ProbeArray.
    return probe(T(x))
end

function Base.promote_rule(::Type{ProbeArray{S,N}}, ::Type{T}) where {S,T<:AbstractScalar,N}
    return promote_type(ProbeArray{S}, T)
end
function Base.promote_rule(::Type{ProbeArray{S}}, ::Type{T}) where {S,T<:AbstractScalar}
    R = promote_type(S, T)
    tensor_type(R) # Fetch tensor type enum to make sure the promoted type is valid.
    return ProbeArray{R}
end
function Base.promote_rule(::Type{ProbeArray{S}}, ::Type{<:AbstractArray{T}}) where {S,T}
    R = promote_type(S, T)
    tensor_type(R) # Fetch tensor type enum to make sure the promoted type is valid.
    return ProbeArray{R}
end
function Base.promote_rule(
    ::Type{ProbeArray{S,N}}, ::Type{<:AbstractArray{T,N}}
) where {S,T,N}
    R = promote_type(S, T)
    tensor_type(R) # Fetch tensor type enum to make sure the promoted type is valid.
    return ProbeArray{R,N}
end
function Base.promote_rule(
    ::Type{ProbeArray{S,M}}, ::Type{<:AbstractArray{T,N}}
) where {S,T,M,N}
    return promote_type(ProbeArray{S}, ProbeArray{T})
end

name(A::ProbeArray) = A.name
raw_size(A::ProbeArray) = A.size
function raw_size(A::ProbeArray, dim)
    dim < 1 && error("arraysize: dimension out of range")
    dim > ndims(A) && return 1
    return A.size[dim]
end

function Base.size(A::ProbeArray)
    return ntuple(dim -> size(A, dim), ndims(A))
end
Base.size(A::ProbeArray, dim) = size(A, Int(dim))
function Base.size(A::ProbeArray, dim::Int)
    d = raw_size(A, dim)
    d isa Int && return d

    start = ndims(A) - dim
    return reshape(onnx_op("Shape", Int64, (1,), A; attr=(start=start, var"end"=start + 1)))
end
function Base.length(A::ProbeArray)
    if raw_size(A) isa Dims
        return prod(raw_size(A))
    end
    return onnx_op("Size", Int64, (), A)
end

# TODO: Is this only needed to override the fallback AbstractArray.
function Base.show(io::IO, A::ProbeArray{T,N}) where {T,N}
    print(io, ProbeArray, "{", T, "}(\"", A.name, "\"")
    for d in raw_size(A)
        print(io, ", ")
        show(io, d)
    end
    print(io, ")")
    return nothing
end
Base.show(io::IO, ::MIME"text/plain", A::ProbeArray) = show(io, A)
Base.iterate(::ProbeArray) = unsupported(iterate)

isprobe(x) = isprobe(typeof(x))
isprobe(::Type{T}) where {T} = false
isprobe(::Type{T}) where {T<:ProbeArray} = true

check_probe(x) = check_probe(typeof(x))
function check_probe(::Type{T}) where {T}
    isprobe(T) && throw(ArgumentError("expected constant but got probe of type $T"))
    return nothing
end

probes(x...) = probe(x)
function probe(x)
    @warn "No probe method for $(typeof(x)), return unchanged value"
    return x
end
probe(A::ProbeArray) = A
probe(t::Tuple) = probe.(t)
function probe(array::AbstractArray, name::String="data")
    @assert !isprobe(array) "Probe types should implement no-op probe methods"
    check_probe(eltype(array))

    # TODO: Check if initializer already exists for the provided array and reuse.
    #       We do not want to store large weights multiple times.

    ctx = GRAPH_CONTEXT[]
    fn = get_value_name(name)
    tensor = TensorProto(array; name=fn)
    push!(ctx.inits, tensor)

    return ProbeArray{eltype(array)}(fn, size(array))
end

function value_info(::Type{T}, dims::ProbeDims, name::String="data") where {T}
    check_probe(T)

    ctx = GRAPH_CONTEXT[]
    fn = get_value_name(name)
    vi = TensorValueInfoProto(fn, T, reverse(dims))
    push!(ctx.values, vi)

    return ProbeArray{T}(fn, dims...)
end

# TODO: Remove?
function to_value_info(A::ProbeArray)
    return TensorValueInfoProto(A.name, eltype(A), reverse(raw_size(A)))
end
function value_info_list(As::Tuple)
    return [to_value_info(A) for A in As]
end
function value_info_list(A)
    return [to_value_info(A)]
end

function unsupported(f)
    return error(
        "Calling ",
        f,
        " on a ProbeArray is not supported. This is intentional and likely the result " *
        "of calling another function that is not yet supported for ONNX export. " *
        "File an issue if you think this is incorrect.",
    )
end
