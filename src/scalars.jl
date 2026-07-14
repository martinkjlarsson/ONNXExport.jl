abstract type AbstractProbeNumber{T} <: Number end

struct ProbeNumber{T} <: AbstractProbeNumber{T}
    name::String

    function ProbeNumber{T}(name::String) where {T}
        check_probe(T)
        return new{T}(name)
    end
end

# TODO: Or should we rely on fill(::ProbeNumber) and (::ProbeScalar)[] instead?
ProbeArray(x::ProbeNumber{T}) where {T} = ProbeArray{T}(name(x))
function ProbeNumber(A::ProbeScalar{T}) where {T}
    return ProbeNumber{T}(name(A))
end

function ProbeNumber{T}(x::ProbeNumber{S}) where {T,S}
    if S == T
        @warn "Converting from ProbeNumber{$S} to ProbeNumber{$T} is a no-op. This should not happen."
        # Perform cast anyway so we create a new ProbeNumber instance.
    end
    return onnx_op("Cast", T, x; attr=(to=Int(tensor_type(T)),))::ProbeNumber{T}
end

function Base.convert(::Type{ProbeNumber{T}}, x::Number) where {T}
    check_probe(x)
    # Convert constant scalar first, then convert to probe.
    return probe(T(x))
end
Base.convert(::Type{T}, x::ProbeNumber) where {T<:ProbeNumber} = x isa T ? x : T(x)::T
# To avoid ambiguities from the two above.
function Base.convert(::Type{ProbeNumber{T}}, x::ProbeNumber) where {T}
    return x isa ProbeNumber{T} ? x : ProbeNumber{T}(x)::ProbeNumber{T}
end

# Allow Int(x) to produce a ProbeNumber{Int}.
(::Type{T})(x::ProbeNumber) where {T<:Number} = convert(ProbeNumber{T}, x)

function Base.promote_rule(::Type{ProbeNumber{S}}, ::Type{T}) where {S,T<:Number}
    # This promotion rule does not apply if T is a probe, return Base.Bottom.
    isprobe(T) && return Base.Bottom

    return ProbeNumber{promote_type(S, T)}
end
function Base.promote_rule(::Type{ProbeNumber{S}}, ::Type{ProbeNumber{T}}) where {S,T}
    return ProbeNumber{promote_type(S, T)}
end

name(p::ProbeNumber) = p.name

raw_size(::ProbeNumber) = ()
function raw_size(::ProbeNumber, dim)
    return dim < 1 ? throw(BoundsError()) : 1
end

isprobe(::Type{T}) where {T<:ProbeNumber} = true

# TODO: Is it an issue that we do not return ProbeNumber{T}?
Base.eltype(::ProbeNumber{T}) where {T} = T
Base.iterate(::ProbeNumber) = unsupported(iterate)

probe(scalar::ProbeNumber, name::String="data") = scalar
function probe(scalar::Number, name::String="data")
    @assert !isprobe(scalar) "Probe types should implement no-op probe methods"

    ctx = GRAPH_CONTEXT[]
    fn = get_value_name(name)
    tensor = TensorProto(scalar; name=fn)
    push!(ctx.inits, tensor)

    return ProbeNumber{typeof(scalar)}(fn)
end

function to_value_info(A::ProbeNumber)
    return TensorValueInfoProto(A.name, eltype(A), [])
end

# TODO: Move somewhere elese?
raw_dims(dims::Tuple) = raw_dim.(dims)
raw_dim(dim::Int) = dim
raw_dim(::ProbeNumber{<:Integer}) = dimension_name()
