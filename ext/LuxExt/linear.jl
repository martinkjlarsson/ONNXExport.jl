function (d::Dense)(x::ProbeArray, ps, st::NamedTuple)
    return with_prefix("Dense") do
        weight = probe(ps.weight, "weight")

        if ndims(x) == 2
            if dynamic(d.use_bias)
                bias = probe(ps.bias, "bias")
                y = d.activation(gemm(weight, x, bias))
            else
                y = d.activation(gemm(weight, x))
            end
        else
            if dynamic(d.use_bias)
                bias = probe(ps.bias, "bias")
                y = d.activation(matmul_onnx(weight, x) .+ bias)
            else
                y = d.activation(matmul_onnx(weight, x))
            end
        end

        return y, st
    end
end
