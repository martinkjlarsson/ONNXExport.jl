function NNlib.pad_reflect(x::ProbeArray, pad::NTuple{M,Int}; dims=1:(M ÷ 2)) where {M}
    return _pad(x, pad, dims, "reflect")
end

function NNlib.pad_reflect(x::ProbeArray, pad::NTuple{2,Int}; dims=1)
    return _pad(x, pad, dims, "reflect")
end

function NNlib.pad_symmetric(x::ProbeArray, pad::NTuple{2,Int}; dims=1)
    @warn "ONNX export of pad_symmetric currently falls back on the implementation in NNlib. This may fail for certain inputs."
    return NNlib._pad_symmetric(x, pad, Val(dims))
end

function NNlib.pad_circular(x::ProbeArray, pad::NTuple{M,Int}; dims=1:(M ÷ 2)) where {M}
    return _pad(x, pad, dims, "wrap")
end

function NNlib.pad_circular(x::ProbeArray, pad::NTuple{2,Int}; dims=1)
    return _pad(x, pad, dims, "wrap")
end

function NNlib.pad_repeat(x::ProbeArray, pad::NTuple{M,Int}; dims=1:(M ÷ 2)) where {M}
    return _pad(x, pad, dims, "edge")
end

function NNlib.pad_repeat(x::ProbeArray, pad::NTuple{2,Int}; dims=1)
    return _pad(x, pad, dims, "edge")
end

function NNlib.pad_constant(
    x::ProbeArray{T,M}, pad::NTuple{N,Tuple{Int,Int}}, val=0
) where {T,M,N}
    return _pad(x, tuplejoin(pad...), 1:M, "constant", val)
end

tuplejoin(x) = x
tuplejoin(x, y) = (x..., y...)
tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)

function _pad(
    x::ProbeArray, pad::NTuple{M,Int}, dims, mode::String, constant=nothing
) where {M}
    new_dims = collect(raw_size(x))
    for (i, d) in enumerate(dims)
        new_dims[d] = ONNXExport.add_dim(new_dims[d], pad[2 * i - 1] + pad[2 * i])
    end
    new_dims = Tuple(new_dims)

    pads = probe(Int64[pad[1:2:end]..., pad[2:2:end]...], "pads")
    if isnothing(constant)
        constant_value = ProbeArray{eltype(x)}("")
    else
        constant_value = probe(eltype(x)(constant), "constant_value")
    end
    axes = probe(Int[ndims(x) .- dims...], "axes")

    return onnx_op("Pad", new_dims, x, pads, constant_value, axes; attr=(mode=mode,))
end
