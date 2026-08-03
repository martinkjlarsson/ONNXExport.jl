# MLUtils defines several intermediate methods such as
# fill_like(x::AbstractArray, val, sz=size(x)) = fill_like(x, val, eltype(x), sz)

function MLUtils.fill_like(::AbstractArray, val::Number, T::Type, dims::ProbeOrIntegers)
    return fill(T(val), dims)
end
function MLUtils.fill_like(x::ProbeArray, val::Number, T::Type=eltype(x))
    ONNXExport.check_probe(val)

    new_dims = raw_size(x)
    if new_dims isa Dims
        # The result is a constant and not a probe.
        return fill(T(val), new_dims)
    else
        shape = onnx_op("Shape", Int64, (ndims(x),), x)
        return onnx_op(
            "ConstantOfShape", T, new_dims, shape; attr=(value=TensorProto([T(val)]),)
        )
    end
end
function MLUtils.fill_like(x::ProbeArray, val::ProbeNumber, T::Type=eltype(x))
    new_dims = raw_size(x)
    if new_dims isa Dims
        shape = probe(collect(Int, reverse(new_dims)))
        return onnx_op("Expand", new_dims, T(val), shape)
    else
        shape = onnx_op("Shape", Int64, (ndims(x),), x)
        return onnx_op("Expand", new_dims, T(val), shape)
    end
end

MLUtils.falses_like(x::ProbeArray) = fill_like(x, false, Bool)
MLUtils.trues_like(x::ProbeArray) = fill_like(x, true, Bool)

MLUtils.zeros_like(x::ProbeArray, T::Type=eltype(x)) = fill_like(x, 0, T)
function MLUtils.zeros_like(x::AbstractArray, T::Type, dims::ProbeOrIntegers)
    return fill_like(x, 0, T, dims)
end

MLUtils.ones_like(x::ProbeArray, T::Type=eltype(x)) = fill_like(x, 1, T)
function MLUtils.ones_like(x::AbstractArray, T::Type, dims::ProbeOrIntegers)
    return fill_like(x, 1, T, dims)
end

MLUtils.rand_like(::AbstractRNG, x::ProbeArray, T::Type, dims::Dims) = rand_like(x, T, dims)
function MLUtils.rand_like(::AbstractRNG, x::AbstractArray, T::Type, dims::ProbeOrIntegers)
    return rand_like(x, T, dims)
end
MLUtils.rand_like(::AbstractRNG, x::ProbeArray, T::Type=eltype(x)) = rand_like(x, T)
MLUtils.rand_like(::ProbeArray, T::Type, dims::Dims) = rand(ProbeRNG(), T, dims)
MLUtils.rand_like(::ProbeArray, T::Type, dims::Integer) = rand(ProbeRNG(), T, dims)
function MLUtils.rand_like(::AbstractArray, T::Type, dims::ProbeOrIntegers)
    return rand(ProbeRNG(), T, dims)
end
function MLUtils.rand_like(x::ProbeArray, T::Type=eltype(x))
    return onnx_op("RandomUniformLike", T, x; attr=(dtype=Int(tensor_type(T)),))
end

function MLUtils.randn_like(::AbstractRNG, x::ProbeArray, T::Type, dims::Dims)
    return randn_like(x, T, dims)
end
function MLUtils.randn_like(::AbstractRNG, x::AbstractArray, T::Type, dims::ProbeOrIntegers)
    return randn_like(x, T, dims)
end
MLUtils.randn_like(::AbstractRNG, x::ProbeArray, T::Type=eltype(x)) = randn_like(x, T)
MLUtils.randn_like(::ProbeArray, T::Type, dims::Dims) = randn(ProbeRNG(), T, dims)
MLUtils.randn_like(::ProbeArray, T::Type, dims::Integer) = randn(ProbeRNG(), T, dims)
function MLUtils.randn_like(::AbstractArray, T::Type, dims::ProbeOrIntegers)
    return randn(ProbeRNG(), T, dims)
end
function MLUtils.randn_like(x::ProbeArray, T::Type=eltype(x))
    return onnx_op("RandomNormalLike", T, x; attr=(dtype=Int(tensor_type(T)),))
end
