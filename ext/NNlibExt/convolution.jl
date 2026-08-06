function NNlib.conv(
    x::ProbeArray,
    w::AbstractArray{T,N};
    stride=1,
    pad=0,
    dilation=1,
    flipped=false,
    groups=1,
) where {T,N}
    return ONNXExport.conv_onnx(
        x,
        w;
        stride=stride,
        pad=pad,
        dilation=dilation,
        cross_correlation=flipped,
        groups=groups,
    )
end

function NNlib.depthwiseconv(
    x::ProbeArray, w::AbstractArray{T,N}; stride=1, pad=0, dilation=1, flipped=false
) where {T,N}
    groups = size(w, N)
    w = reshape(w, size(w)[1:(N - 2)]..., 1, size(w, N - 1) * size(w, N))
    return ONNXExport.conv_onnx(
        x,
        w;
        stride=stride,
        pad=pad,
        dilation=dilation,
        cross_correlation=flipped,
        groups=groups,
    )
end
