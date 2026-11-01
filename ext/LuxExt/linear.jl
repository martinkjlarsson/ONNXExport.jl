function (d::Dense)(x::ProbeArray, ps, st::NamedTuple)
    return ONNXExport.with_prefix("Dense") do
        weight = probe(ps.weight, "weight")

        if ndims(x) == 2
            if dynamic(d.use_bias)
                bias = probe(ps.bias, "bias")
                y = d.activation(ONNXExport.gemm(weight, x, bias))
            else
                y = d.activation(ONNXExport.gemm(weight, x))
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

# These Bilinear implementations are only needed to avoid asserts.
function (b::Bilinear)((x, y)::Tuple{<:AbstractArray,<:ProbeArray}, ps, st::NamedTuple)
    return _bilinear(b, (x, y), ps, st)
end
function (b::Bilinear)((x, y)::Tuple{<:ProbeArray,<:AbstractArray}, ps, st::NamedTuple)
    return _bilinear(b, (x, y), ps, st)
end
function (b::Bilinear)((x, y)::Tuple{<:ProbeArray,<:ProbeArray}, ps, st::NamedTuple)
    return _bilinear(b, (x, y), ps, st)
end

function _bilinear(
    b::Bilinear, (x, y)::Tuple{<:ProbeVecOrMat,<:ProbeVecOrMat}, ps, st::NamedTuple
)
    s₁, s₂, s₃ = size(ps.weight)

    Wy = reshape(reshape(ps.weight, (:, s₃)) * y, (s₁, s₂, :))
    Wyx = reshape(matmul_onnx(Wy, reshape(x, (s₂, 1, :))), (s₁, :))

    if dynamic(b.use_bias)
        bias = probe(ps.bias, "bias")
        z = Wyx .+ bias
    else
        z = Wyx
    end

    z = b.activation(z)

    return z, st
end

function _bilinear(
    b::Bilinear, (x, y)::Tuple{<:ProbeArray,<:ProbeArray}, ps, st::NamedTuple
)
    s₁, s₂, s₃ = size(ps.weight)
    x′ = reshape(x, s₂, :)
    y′ = reshape(y, s₃, :)

    z, stₙ = _bilinear(b, (x′, y′), ps, st)

    return reshape(z, s₁, size(x)[2:end]...), stₙ
end
