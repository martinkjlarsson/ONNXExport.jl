function NNlib.maxpool(x::ProbeArray{T,N}, pdims::PoolDims) where {N,T}
    new_dims = pool_output_size(x, pdims)
    if pool_is_global(pdims)
        return onnx_op("GlobalMaxPool", new_dims, x)
    else
        attr = pool_attr(pdims)
        return onnx_op("MaxPool", new_dims, x; attr=attr)
    end
end

function NNlib.meanpool(x::ProbeArray{T,N}, pdims::PoolDims) where {N,T}
    new_dims = pool_output_size(x, pdims)
    if pool_is_global(pdims)
        return onnx_op("GlobalAveragePool", new_dims, x)
    else
        attr = pool_attr(pdims; count_include_pad=1)
        return onnx_op("AveragePool", new_dims, x; attr=attr)
    end
end

function NNlib.lpnormpool(x::ProbeArray{T,N}, pdims::PoolDims; p::Real) where {N,T}
    @assert isinteger(p) "ONNX export of Lp-pooling requires integer p"

    new_dims = pool_output_size(x, pdims)
    if pool_is_global(pdims)
        return onnx_op("GlobalLpPool", new_dims, x; attr=(p=p,))
    else
        attr = pool_attr(pdims; p=p)
        return onnx_op("LpPool", new_dims, x; attr=attr)
    end
end

function pool_output_size(x, pdims)
    return NNlib.output_size(pdims)..., NNlib.channels_out(pdims), NNlib.size(x, ndims(x))
end

pool_is_global(pdims) = NNlib.input_size(pdims) == NNlib.kernel_size(pdims)

function pool_attr(pdims; kwargs...)
    pads = (NNlib.padding(pdims)[(end - 1):-2:1]..., NNlib.padding(pdims)[end:-2:1]...)
    attr = (
        dilations=reverse(NNlib.dilation(pdims)),
        kernel_shape=reverse(NNlib.kernel_size(pdims)),
        pads=pads,
        strides=reverse(NNlib.stride(pdims)),
        kwargs...,
    )
    return attr
end
