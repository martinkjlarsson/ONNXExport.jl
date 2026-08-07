struct ProbeRNG <: AbstractRNG end

# TODO: Could this be done less verbose with Sampler or similar?
Base.rand(::ProbeRNG, ::Type{T}, dims::Dims) where {T} = _randu(T, dims)
Base.rand(::ProbeRNG, ::Type{T}, dims::ProbeOrIntegers) where {T} = _randu(T, dims)
Base.rand(::ProbeRNG, ::Type{T}, dims::ProbeOrInteger...) where {T} = _randu(T, dims)
Base.rand(::ProbeRNG, ::Type{T}, dims::Integer...) where {T} = _randu(T, dims)
Base.rand(::ProbeRNG, dims::Integer...) = _randu(Float64, dims)
Base.rand(::ProbeRNG, dims::ProbeOrInteger...) = _randu(Float64, dims)

Base.randn(::ProbeRNG, T::Random.BitFloatType) = _randn(T, ())
Base.randn(::ProbeRNG, ::Type{T}, dims::Dims) where {T} = _randn(T, dims)
Base.randn(::ProbeRNG, ::Type{T}, dims::ProbeOrIntegers) where {T} = _randn(T, dims)
Base.randn(::ProbeRNG, ::Type{T}, dims::ProbeOrInteger...) where {T} = _randn(T, dims)
Base.randn(::ProbeRNG, ::Type{T}, dims::Integer...) where {T} = _randn(T, dims)
Base.randn(::ProbeRNG, dims::Integer...) = _randn(Float64, dims)
Base.randn(::ProbeRNG, dims::ProbeOrInteger...) = _randn(Float64, dims)

Random.randexp(::ProbeRNG, T::Random.BitFloatType) = _rande(T, ())
Random.randexp(::ProbeRNG, ::Type{T}, dims::Dims) where {T} = _rande(T, dims)
Random.randexp(::ProbeRNG, ::Type{T}, dims::ProbeOrIntegers) where {T} = _rande(T, dims)
Random.randexp(::ProbeRNG, ::Type{T}, dims::ProbeOrInteger...) where {T} = _rande(T, dims)
Random.randexp(::ProbeRNG, ::Type{T}, dims::Integer...) where {T} = _rande(T, dims)
Random.randexp(::ProbeRNG, dims::Integer...) = _rande(Float64, dims)
Random.randexp(::ProbeRNG, dims::ProbeOrInteger...) = _rande(Float64, dims)

Random.bitrand(::ProbeRNG, dims::Dims) = _randb(dims)
Random.bitrand(::ProbeRNG, dims::ProbeOrIntegers) = _randb(dims)
Random.bitrand(::ProbeRNG, dims::Integer...) = _randb(dims)
Random.bitrand(::ProbeRNG, dims::ProbeOrInteger...) = _randb(dims)

_randu(T, dims) = _random("RandomUniform", "RandomUniformLike", T, dims)
_randn(T, dims) = _random("RandomNormal", "RandomNormalLike", T, dims)
_rande(T, dims) = -log.(one(T) .- _randu(T, dims))
_randb(dims) = _randu(Bool, dims)

function _random(dist::String, distlike::String, ::Type{T}, dims::Tuple) where {T}
    new_dims = map(d -> isprobe(d) ? dimension_name() : Int(d), dims)
    if !isa(new_dims, Dims)
        # One of dims is a probe, might as well convert all dims to probe before vcat.
        shape = vcat(probe(reverse(dims))...)
        ref = onnx_op(
            "ConstantOfShape", T, new_dims, shape; attr=(value=TensorProto([zero(T)]),)
        )
        return onnx_op(distlike, ref)
    elseif T <: AbstractFloat
        # RandomUniform and RandomNormal only support float types.
        attr = (dtype=Int(tensor_type(T)), shape=collect(Int, reverse(new_dims)))
        return onnx_op(dist, T, new_dims; attr=attr)
    else
        ref = probe(zeros(T, new_dims))
        return onnx_op(distlike, ref)
    end
end
