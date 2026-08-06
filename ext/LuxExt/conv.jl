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
