function (f::FlattenLayer{Nothing})(x::ProbeArray{T,N}, _, st::NamedTuple) where {T,N}
    return flatten(x), st
end

function (f::FlattenLayer)(x::ProbeArray{T,N}, _, st::NamedTuple) where {T,N}
    f.N == N-1 && return flatten(x), st

    return reshape(x, :, size(x)[(f.N + 1):end]...), st
end

function flatten(x)
    sz = raw_size(x)
    sz1 = sz[1:(end - 1)]
    sz2 = sz[end]

    new_dims = sz1 isa Dims ? (prod(sz1), sz2) : (dimension_name(), sz2)
    return onnx_op("Flatten", new_dims, x)
end
