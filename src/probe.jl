"""
    name(x)

Return the name of the probe `x`, corresponding to the ONNX tensor name.
"""
function name end

"""
    raw_size(x, [dim])

Return the dimensions of the underlying ONNX tensor `x` represents.

Return a tuple of `Union{Int,Symbol}`, where a `Symbol` indicates the name of a
symbolic dimension. Note that `size(x)` will likely return a tuple of
`Union{Int,ProbeNumber{Int}}` correspondingly. Optionally you can specify a dimension to
just get the length/symbol of that dimension.
"""
function raw_size end
function raw_size(x, dim)
    dim < 1 && throw(BoundsError())
    dims = raw_size(x)
    dim > length(dims) && return 1
    return dims[dim]
end

"""
    isprobe(x)
    isprobe(T::Type)

Return `true` if `x` is a probe.
"""
isprobe(x) = isprobe(typeof(x))
isprobe(::Type{T}) where {T} = false

"""
    probe(x, [name])

Convert `x` to a probe if possible.

If `x` is a value that can be represented as an ONNX tensor, an ONNX initializer is created
and a suitable probe value is returned. Optionally, a name can be given to the ONNX tensor.
The name may be changed with prefixes or suffixes to ensure uniqueness. If `x` already is
a probe or cannot be represented as an ONNX tensor, `x` is returned unchanged.
"""
function probe(x, ::String="")
    @warn "No probe method for type $(typeof(x)), returning unchanged value. To get rid of this warning, define probe(x::$(typeof(x)))."
    return x
end
probe(t::Tuple) = probe.(t) # TODO: Is this used?

check_probe(x) = check_probe(typeof(x))
function check_probe(::Type{T}) where {T}
    isprobe(T) && throw(ArgumentError("expected constant but got probe of type $T."))
    return nothing
end

const ProbeDim = Union{Int,Symbol}
const ProbeDims{N} = NTuple{N,ProbeDim}

mul_dim(a::Int, b::Int) = a * b
mul_dim(::ProbeDim, ::ProbeDim) = dimension_name()
div_dim(a::Int, b::Int) = a ÷ b
div_dim(::ProbeDim, ::ProbeDim) = dimension_name()
add_dim(a::Int, b::Int) = a + b
add_dim(::ProbeDim, ::ProbeDim) = dimension_name()

const ExactlyOne{S,T} = Union{
    Tuple{S,Vararg{T}},Tuple{T,S,Vararg{T}},Tuple{T,T,S,Vararg{T}},Tuple{T,T,T,S,Vararg{T}}
}
const AtLeastOne{S,T} = ExactlyOne{S,Union{S,T}}
