"""
Represents a `ProbeArray` in a broadcasted call. `BroadcastProbe` behaves like a `Number`
but produces ONNX operators performing elementwise operations on the underlying
`ProbeArray`.
"""
struct BroadcastProbe{T} <: AbstractProbeNumber{T}
    probe::ProbeArray{T}
end

# TODO: This may cause errors with, e.g., a .+ b, where a and b are scalars results in a ProbeArray.
#       It might be better to make BroadcastProbe{T,P} where P = ProbeArray{T} or P = ProbeNumber{T}
BroadcastProbe(x::ProbeNumber{T}) where {T} = BroadcastProbe(ProbeArray(x))

function BroadcastProbe{T}(A::BroadcastProbe{S}) where {S,T}
    return BroadcastProbe(convert(ProbeArray{T}, A.probe))
end

function Base.promote_rule(::Type{BroadcastProbe{S}}, ::Type{T}) where {S,T<:Number}
    check_probe(T)
    return BroadcastProbe{promote_type(S, T)}
end
function Base.promote_rule(::Type{BroadcastProbe{S}}, ::Type{ProbeNumber{T}}) where {S,T}
    return BroadcastProbe{promote_type(S, T)}
end
function Base.promote_rule(::Type{BroadcastProbe{S}}, ::Type{BroadcastProbe{T}}) where {S,T}
    return BroadcastProbe{promote_type(S, T)}
end

function Base.convert(::Type{BroadcastProbe{T}}, x::Number) where {T}
    check_probe(x)
    return BroadcastProbe(probe(T(x)))::BroadcastProbe{T}
end
function Base.convert(::Type{BroadcastProbe{S}}, x::ProbeNumber{T}) where {S,T}
    return BroadcastProbe(convert(ProbeNumber{S}, x))::BroadcastProbe{S}
end
function Base.convert(::Type{BroadcastProbe{T}}, A::BroadcastProbe) where {T}
    return A isa BroadcastProbe{T} ? A : BroadcastProbe{T}(A)::BroadcastProbe{T}
end
# Allow Int(x) to produce a BroadcastProbe{Int}.
(::Type{T})(x::BroadcastProbe) where {T<:Number} = convert(BroadcastProbe{T}, x)

Base.eltype(::BroadcastProbe{T}) where {T} = T

name(A::BroadcastProbe) = name(A.probe)
raw_size(A::BroadcastProbe) = raw_size(A.probe)

isprobe(::Type{T}) where {T<:BroadcastProbe} = true

unwrap_broadcast(x) = x
unwrap_broadcast(A::BroadcastProbe) = A.probe
unwrap_broadcast(A::Tuple) = unwrap_broadcast.(A)
wrap_broadcast(x) = x # TODO: Should we always return x[]?
wrap_broadcast(x::Base.RefValue) = x[]
wrap_broadcast(A::ProbeArray) = BroadcastProbe(A)
wrap_broadcast(A::ProbeNumber) = A # TODO: Or wrap with BroadcastProbe?
wrap_broadcast(A::Tuple) = wrap_broadcast.(A)

function Base.broadcastable(::BroadcastProbe)
    # TODO: This might be allowed after all.
    return error("Nested broadcasting is not supported by ProbeArray")
end
Base.broadcastable(A::ProbeArray) = A
struct ProbeStyle <: Base.BroadcastStyle end
Base.BroadcastStyle(::Type{<:ProbeArray}) = ProbeStyle()
Base.BroadcastStyle(::Type{<:ProbeNumber}) = ProbeStyle()
Base.BroadcastStyle(::ProbeStyle, ::Base.BroadcastStyle) = ProbeStyle()

function Base.broadcasted(::ProbeStyle, f, args...)
    args = probe.(args)
    pbs = wrap_broadcast(args)
    results = f(pbs...)
    return unwrap_broadcast(results)
end

# Hopefully, these constants will be promoted to ProbeArray if needed.
Base.zero(::Type{BroadcastProbe{T}}) where {T} = zero(T)
Base.one(::Type{BroadcastProbe{T}}) where {T} = one(T)
# TODO: What about typemax and similar functions?
# TODO: Should it be BroadcastProbe or AbstractProbeNumber?

function broadcast_shape(As...)
    return broadcast_shape(raw_size.(As)...)
end
function broadcast_shape(a::ProbeDims, b::ProbeDims, cs::ProbeDims...)
    return broadcast_shape(a, broadcast_shape(b, cs...))
end
function broadcast_shape(a::ProbeDims, b::ProbeDims)
    return ntuple(max(length(a), length(b))) do i
        merge_dim(i <= length(a) ? a[i] : 1, i <= length(b) ? b[i] : 1)
    end
end
broadcast_shape(a::ProbeDims) = a

function merge_dim(a::Int, b::Int)
    a == 1 && return b
    b == 1 && return a
    a == b && return a
    throw(
        DimensionMismatch(
            LazyString(
                "arrays could not be broadcast to a common size; got a dimension with lengths ",
                a,
                " and ",
                b,
            ),
        ),
    )
end
merge_dim(a::Int, b::Symbol) = merge_dim(b, a)
function merge_dim(a::Symbol, b::Int)
    b == 1 && return a
    @warn "Dimensions do not match, assuming dynamic dimension $a=$b"
    return b
end
function merge_dim(a::Symbol, b::Symbol)
    if a != b
        @warn "Dimensions do not match, assuming dynamic dimensions $a and $b are equal"
    end
    return a
end
