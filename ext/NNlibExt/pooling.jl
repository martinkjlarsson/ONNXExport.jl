function NNlib.maxpool(x::ProbeArray{T,N}, pdims::PoolDims) where {N,T}
    if NNlib.input_size(pdims) == NNlib.kernel_size(pdims)
        new_dims = (ntuple(Returns(1), N - 2)..., raw_size(x, N - 1), raw_size(x, N))
        return onnx_op("GlobalMaxPool", new_dims, x)
    else
        error("pdims = $pdims")
        # TODO: Complicated stuff with MaxPool
        # return onnx_op("MaxPool", new_dims, x)
    end
end

function NNlib.meanpool(x::ProbeArray{T,N}, pdims::PoolDims) where {N,T}
    if NNlib.input_size(pdims) == NNlib.kernel_size(pdims)
        new_dims = (ntuple(Returns(1), N - 2)..., raw_size(x, N - 1), raw_size(x, N))
        return onnx_op("GlobalAveragePool", new_dims, x)
    else
        error("pdims = $pdims")
        # TODO: Complicated stuff with AveragePool
        # return onnx_op("AveragePool", new_dims, x)
    end
end

function NNlib.lpnormpool(x::ProbeArray{T,N}, pdims::PoolDims; p::Real) where {N,T}
    @assert isinteger(p) "ONNX export of Lp-pooling requires integer p"
    if NNlib.input_size(pdims) == NNlib.kernel_size(pdims)
        new_dims = (ntuple(Returns(1), N - 2)..., raw_size(x, N - 1), raw_size(x, N))
        return onnx_op("GlobalLpPool", new_dims, x; attr=(p=p,))
    else
        error("pdims = $pdims")
        # TODO: Complicated stuff with LpPool
        # return onnx_op("LpPool ", new_dims, x)
    end
end
