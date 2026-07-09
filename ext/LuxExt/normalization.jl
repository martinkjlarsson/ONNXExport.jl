function (l::LayerNorm)(x::ProbeArray, ps, st::NamedTuple)
    return with_prefix("LayerNorm") do
        last_dim = layernorm_dim(l.dims, ndims(x))
        attr = (axis=ndims(x) - last_dim, epsilon=l.epsilon)

        # TODO: Should stash_type attribute be set to Int(tensor_type(eltype(x)))?

        # Drop trailing singleton dimensions to avoid bug in ONNX runtime caused by the
        # SkipLayerNormalization fusion optimization. This should be fixed after 1.24.2.
        # See the issue https://github.com/microsoft/onnxruntime/issues/27455.
        function droptrailing(x)
            while ndims(x) > 0 && size(x, ndims(x)) == 1
                x = dropdims(x; dims=ndims(x))
            end
            return x
        end

        if dynamic(l.affine)
            c1 = prod(l.shape)
            c2 = prod(size(x)[1:last_dim])
            if c1 != c2
                error(
                    "ONNX LayerNormalization requires prod(shape) = length(x[dims]). Got prod($(l.shape)) = $c1 != $c2 = prod($(size(x)[1:last_dim])).",
                )
            end

            scale = probe(droptrailing(ps.scale), "scale")
            bias = probe(droptrailing(ps.bias), "bias")
            y = onnx_op("LayerNormalization", x, scale, bias; attr=attr)
        else
            # Scale is mandatory. Bias is optional.
            psscale = ones(eltype(x), l.shape..., 1)
            scale = probe(droptrailing(psscale), "scale")
            y = onnx_op("LayerNormalization", x, scale; attr=attr)
        end

        y = l.activation(y)

        return y, st
    end
end

layernorm_dim(::Colon, n) = n
function layernorm_dim(dims::Integer, n)
    dims != 1 && layernorm_dim(nothing, n)
    return dims
end
function layernorm_dim(dims::AbstractUnitRange{<:Integer}, n)
    first(dims) != 1 && layernorm_dim(nothing, n)
    return last(dims)
end
function layernorm_dim(::Any, _)
    return error(
        "The ONNX LayerNormalization operator only supports normalization over the first dimensions 1:i for some i in 1..ndims(x).",
    )
end
