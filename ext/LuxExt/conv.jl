function (c::Conv)(x::ProbeArray, ps, st::NamedTuple)
    return with_prefix("Conv") do
        # ONNX padding has the format (begin2, begin1, end2, end1) while Lux uses
        # (begin1, end1, begin2, end2), where dimension 1 is the fastest changing
        # dimension. The padding is not affected by c.cross_correlation.
        pads = (c.pad[(end - 1):-2:1]..., c.pad[end:-2:1]...)

        attr = (
            dilations=reverse(c.dilation),
            group=c.groups,
            kernel_shape=reverse(size(ps.weight)[1:(end - 2)]),
            pads=pads,
            strides=reverse(c.stride),
        )

        weight = ps.weight
        if !dynamic(c.cross_correlation)
            # ONNX convolution is actually cross correlation.
            weight = reverse(weight; dims=ntuple(identity, ndims(ps.weight) - 2))
        end
        weight = probe(weight, "weight")

        # See NNlib/src/dim_helpers/ConvDims.jl:34
        output_dims = ntuple(ndims(x) - 2) do i
            pad_dil =
                c.pad[(i - 1) * 2 + 1] + c.pad[(i - 1) * 2 + 2] -
                (size(weight, i) - 1) * c.dilation[i] - 1
            if pad_dil == -1 && c.stride[i] == 1
                # Input and output size are the same. Keep any symbolic dimension.
                return raw_size(x, i)
            elseif raw_size(x, i) isa Symbol
                return dimension_name()
            else
                return div(raw_size(x, i) + pad_dil, c.stride[i]) + 1
            end
        end
        new_dims = (output_dims..., c.out_chs, raw_size(x, ndims(x)))

        if dynamic(c.use_bias)
            bias = probe(ps.bias, "bias")
            @assert eltype(x) == eltype(weight) == eltype(bias) "Implicit type promotion not yet supported"
            y = onnx_op("Conv", new_dims, x, weight, bias; attr=attr)
        else
            @assert eltype(x) == eltype(weight) "Implicit type promotion not yet supported"
            y = onnx_op("Conv", new_dims, x, weight; attr=attr)
        end

        y = c.activation(y)

        return y, st
    end
end
