function (c::Conv)(x::ProbeArray, ps, st::NamedTuple)
    return ONNXExport.with_prefix("Conv") do
        bias = dynamic(c.use_bias) ? ps.bias : nothing
        y = ONNXExport.conv_onnx(
            x,
            ps.weight,
            bias;
            stride=c.stride,
            pad=c.pad,
            dilation=c.dilation,
            cross_correlation=dynamic(c.cross_correlation),
            groups=c.groups,
        )

        y = c.activation(y)

        return y, st
    end
end

function (c::ConvTranspose)(x::ProbeArray, ps, st::NamedTuple)
    return ONNXExport.with_prefix("ConvTranspose") do
        weight = ps.weight
        n_spatial = ndims(x) - 2

        stride = ONNXExport.expand_tuple(c.stride, Val(n_spatial))
        pad = ONNXExport.expand_tuple(c.pad, Val(2 * n_spatial))
        dilation = ONNXExport.expand_tuple(c.dilation, Val(n_spatial))
        output_padding = ONNXExport.expand_tuple(c.outpad, Val(n_spatial))

        # Lux stores each spatial dimension's begin/end padding consecutively,
        # whereas ONNX groups the begin and end pads and orders dimensions backwards.
        pads = (pad[(end - 1):-2:1]..., pad[end:-2:1]...)
        attr = (
            dilations=reverse(dilation),
            group=c.groups,
            kernel_shape=reverse(size(weight)[1:n_spatial]),
            output_padding=reverse(output_padding),
            pads=pads,
            strides=reverse(stride),
        )

        if !dynamic(c.cross_correlation)
            weight = reverse(weight; dims=ntuple(identity, n_spatial))
        end

        output_spatial_dims = ntuple(n_spatial) do i
            offset =
                1 - stride[i] + (size(weight, i) - 1) * dilation[i] - pad[2 * i - 1] -
                pad[2 * i] + output_padding[i]
            input_dim = raw_size(x, i)
            if stride[i] == 1 && offset == 0
                return input_dim
            elseif input_dim isa Symbol
                return ONNXExport.dimension_name()
            else
                return stride[i] * input_dim + offset
            end
        end
        output_dims = (
            output_spatial_dims...,
            size(weight, ndims(weight) - 1) * c.groups,
            raw_size(x, ndims(x)),
        )

        bias = dynamic(c.use_bias) ? ps.bias : nothing
        if isnothing(bias)
            @assert eltype(x) == eltype(weight) "Implicit type promotion not yet supported"
        else
            @assert eltype(x) == eltype(weight) == eltype(bias) "Implicit type promotion not yet supported"
        end
        weight = probe(weight, "weight")
        if isnothing(bias)
            y = onnx_op("ConvTranspose", output_dims, x, weight; attr=attr)
        else
            bias = probe(bias, "bias")
            y = onnx_op("ConvTranspose", output_dims, x, weight, bias; attr=attr)
        end

        y = c.activation(y)

        return y, st
    end
end
