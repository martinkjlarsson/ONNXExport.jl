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

function NNlib.unfold(
    x::ProbeArray{T,N}, kernel_size::NTuple{K}; stride=1, pad=0, dilation=1, flipped=true
) where {T,K,N}
    return error(
        "ONNX export of unfold is not supported as there is no corresponding ONNX " *
        "operator. Export with static spatial dimensions could be implemented using " *
        "primitive ONNX operators. File an issue if this is important for your use case.",
    )
end

function NNlib.fold(
    x::ProbeArray{T,3},
    output_size::NTuple{N},
    kernel_size::NTuple{K};
    stride=1,
    pad=0,
    dilation=1,
    flipped=true,
) where {T,K,N}
    if !flipped
        spatial_dims = prod(kernel_size[1:(N - 2)])
        input_channels = kernel_size[N - 1]
        z = reshape(x, size(x, 1), spatial_dims, input_channels, size(x, 3))
        z = reverse(z; dims=2)
        x = ONNXExport.reshape_like(z, x)
    end

    stride = ONNXExport.expand_tuple(stride, Val(N-2))
    pad = ONNXExport.expand_tuple(pad, Val(2*(N-2)))
    dilation = ONNXExport.expand_tuple(dilation, Val(N-2))

    if length(pad) == N-2
        pad = ONNXExport.interleave_tuples(pad, pad)
    end

    # ONNX padding has the format (begin2, begin1, end2, end1) while Lux uses
    # (begin1, end1, begin2, end2), where dimension 1 is the fastest changing
    # dimension. The padding is not affected by cross_correlation.
    pad_onnx = (pad[(end - 1):-2:1]..., pad[end:-2:1]...)

    attr = (dilations=reverse(dilation), pads=pad_onnx, strides=reverse(stride))

    new_dims = ONNXExport.raw_dims(output_size)
    image_shape = probe(vcat(reverse(output_size[1:(N - 2)])...), "image_shape")
    block_shape = probe(vcat(reverse(kernel_size[1:(N - 2)])...), "block_shape")

    return onnx_op("Col2Im", new_dims, x, image_shape, block_shape; attr=attr)
end
