function conv_onnx(
    x::ProbeArray{T,N},
    weight::AbstractArray{T,N},
    bias::Union{Nothing,AbstractVector{T}}=nothing;
    stride=1,
    pad=0,
    dilation=1,
    cross_correlation=true,
    groups=1,
) where {T,N}
    stride = expand_tuple(stride, Val(N-2))
    pad = expand_tuple(pad, Val(2*(N-2)))
    dilation = expand_tuple(dilation, Val(N-2))

    if length(pad) == N-2
        pad = interleave_tuples(pad, pad)
    end

    # ONNX padding has the format (begin2, begin1, end2, end1) while Lux uses
    # (begin1, end1, begin2, end2), where dimension 1 is the fastest changing
    # dimension. The padding is not affected by cross_correlation.
    pad_onnx = (pad[(end - 1):-2:1]..., pad[end:-2:1]...)

    attr = (
        dilations=reverse(dilation),
        group=groups,
        kernel_shape=reverse(size(weight)[1:(end - 2)]),
        pads=pad_onnx,
        strides=reverse(stride),
    )

    if !cross_correlation
        # ONNX convolution is actually cross correlation.
        weight = reverse(weight; dims=ntuple(identity, N - 2))
    end

    # See NNlib/src/dim_helpers/ConvDims.jl:34
    output_dims = ntuple(N - 2) do i
        pad_dil =
            pad[(i - 1) * 2 + 1] + pad[(i - 1) * 2 + 2] -
            (size(weight, i) - 1) * dilation[i] - 1
        if pad_dil == -1 && stride[i] == 1
            # Input and output size are the same. Keep any symbolic dimension.
            return raw_size(x, i)
        elseif raw_size(x, i) isa Symbol
            return dimension_name()
        else
            return div(raw_size(x, i) + pad_dil, stride[i]) + 1
        end
    end
    new_dims = (output_dims..., size(weight, N), raw_size(x, N))

    if isnothing(bias)
        @assert eltype(x) == eltype(weight) "Implicit type promotion not yet supported"
        weight = probe(weight, "weight")
        y = onnx_op("Conv", new_dims, x, weight; attr=attr)
    else
        @assert eltype(x) == eltype(weight) == eltype(bias) "Implicit type promotion not yet supported"
        weight = probe(weight, "weight")
        bias = probe(bias, "bias")
        y = onnx_op("Conv", new_dims, x, weight, bias; attr=attr)
    end

    return y
end
