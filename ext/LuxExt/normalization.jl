function (BN::BatchNorm)(x::ProbeArray, ps, st::NamedTuple)
    training_mode = st.training isa Val{true}
    if !training_mode && BN.track_stats isa False
        @warn "ONNX export of BatchNorm with training=false and track_stats=false is " *
            "not possible as the BatchNormalization operator requires mean and variance " *
            "inputs. Exporting as if training=true."
        training_mode = true
    end

    if training_mode
        @warn "Due to an issue in ONNXRunTime, exported models with training=true might not give correct results."
    end

    attr = (epsilon=BN.epsilon, momentum=1-BN.momentum, training_mode=training_mode)

    if BN.affine isa True
        scale = ps.scale
        bias = ps.bias
    else
        scale = ones(eltype(x), BN.chs)
        bias = zeros(eltype(x), BN.chs)
    end

    if BN.track_stats isa True
        input_mean = st.running_mean
        input_var = st.running_var
    else
        # Can be whatever as we do not use the outputs from BatchNormalization.
        input_mean = zeros(eltype(x), BN.chs)
        input_var = ones(eltype(x), BN.chs)
    end

    scale = probe(scale, "scale")
    bias = probe(bias, "bias")
    input_mean = probe(input_mean, "input_mean")
    input_var = probe(input_var, "input_var")

    if training_mode
        y = value_info(eltype(x), raw_size(x), "y")
        running_mean = value_info(eltype(input_mean), (BN.chs,), "running_mean")
        running_var = value_info(eltype(input_var), (BN.chs,), "running_var")
        onnx_op(
            "BatchNormalization",
            (x, scale, bias, input_mean, input_var),
            (y, running_mean, running_var);
            attr=attr,
        )

        if BN.track_stats isa True
            st2 = merge(st, (running_mean=running_mean, running_var=running_var))
        else
            st2 = st
        end
    else
        y = onnx_op(
            "BatchNormalization",
            raw_size(x),
            x,
            scale,
            bias,
            input_mean,
            input_var;
            attr=attr,
        )

        st2 = st
    end

    y = BN.activation(y)

    return y, st2
end

function (GN::GroupNorm)(x::ProbeArray, ps, st::NamedTuple)
    attr = (epsilon=GN.epsilon, num_groups=GN.groups)

    if GN.affine isa True
        scale = ps.scale
        bias = ps.bias
    else
        scale = ones(eltype(x), GN.chs)
        bias = zeros(eltype(x), GN.chs)
    end

    scale = probe(scale, "scale")
    bias = probe(bias, "bias")

    y = onnx_op("GroupNormalization", raw_size(x), x, scale, bias; attr=attr)
    y = GN.activation(y)

    return y, st
end

function (IN::InstanceNorm)(x::ProbeArray, ps, st::NamedTuple)
    if IN.track_stats isa True
        @warn "ONNX export of InstanceNorm with track_stats=true is not supported. Assuming track_stats=false."
    end

    attr = (epsilon=IN.epsilon,)

    if IN.affine isa True
        scale = ps.scale
        bias = ps.bias
    else
        scale = ones(eltype(x), IN.chs)
        bias = zeros(eltype(x), IN.chs)
    end

    scale = probe(scale, "scale")
    bias = probe(bias, "bias")

    y = onnx_op("InstanceNormalization", raw_size(x), x, scale, bias; attr=attr)
    y = IN.activation(y)

    return y, st
end

function (l::LayerNorm)(x::ProbeArray, ps, st::NamedTuple)
    return with_prefix("LayerNorm") do
        last_dim = layernorm_dim(l.dims, ndims(x))
        attr = (axis=ndims(x) - last_dim, epsilon=l.epsilon)

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
            y = onnx_op("LayerNormalization", raw_size(x), x, scale, bias; attr=attr)
        else
            # Scale is mandatory. Bias is optional.
            psscale = ones(eltype(x), l.shape..., 1)
            scale = probe(droptrailing(psscale), "scale")
            y = onnx_op("LayerNormalization", raw_size(x), x, scale; attr=attr)
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

function (rms::RMSNorm)(x::ProbeArray, ps, st::NamedTuple)
    N = length(rms.normalized_shape)

    # TODO: RMSNormalization requires opset 23. Add preference/option for opset.
    # attr = (epsilon=rms.epsilon, axis=ndims(x) - N)

    # if rms.affine isa True
    #     scale = ps.scale
    # else
    #     scale = ones(eltype(x), rms.normalized_shape)
    # end

    # scale = probe(scale, "scale")
    # y = onnx_op("RMSNormalization", raw_size(x), x, scale; attr=attr)

    rms_val = sqrt.(mean(abs2, x; dims=1:N) .+ rms.epsilon)

    if rms.affine isa True
        y = x ./ rms_val .* ps.scale
    else
        y = x ./ rms_val
    end

    return y, st
end
