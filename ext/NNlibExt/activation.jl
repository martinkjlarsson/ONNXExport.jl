function act_fun(op_type, x::AbstractProbeNumber{T}; attr...) where {T<:AbstractFloat}
    return onnx_op(op_type, x; attr=attr)
end
function act_fun(op_type, x::AbstractProbeNumber{T}; attr...) where {T}
    return onnx_op(op_type, float(T)(x); attr=attr)
end

NNlib.celu(x::AbstractProbeNumber, α=1) = act_fun("Celu", x; alpha=Float32(α))

NNlib.elu(x::AbstractProbeNumber, α=1) = act_fun("Elu", x; alpha=Float32(α))

# gelu delegates to gelu_tanh(x)
# NNlib.gelu(x::AbstractProbeNumber)

NNlib.gelu_tanh(x::AbstractProbeNumber) = act_fun("Gelu", x; approximate="tanh")

NNlib.gelu_sigmoid(x::AbstractProbeNumber) = NNlib.gelu_tanh(x)

NNlib.gelu_erf(x::AbstractProbeNumber) = act_fun("Gelu", x)

# max(0, min(1, (x + 3) / 6)) = max(0, min(1, alpha * x + beta))
function NNlib.hardsigmoid(x::AbstractProbeNumber)
    return act_fun("HardSigmoid", x; alpha=1.0f0 / 6.0f0, beta=0.5f0)
end

NNlib.hardtanh(x::AbstractProbeNumber) = clamp(x, -1, 1)

NNlib.leakyrelu(x::AbstractProbeNumber, a=0.01) = act_fun("LeakyRelu", x; alpha=Float32(a))

# Implemented in NNlib.
# NNlib.lisht(x::AbstractProbeNumber) = x * tanh(x)

NNlib.logcosh(x::AbstractProbeNumber{T}) where {T} = x + softplus(-2x) - float(T)(log(2))

# Implemented in NNlib.
# NNlib.logsigmoid(x::AbstractProbeNumber) = -softplus(-x)

NNlib.mish(x::AbstractProbeNumber) = act_fun("Mish", x)

NNlib.relu(x::AbstractProbeNumber) = act_fun("Relu", x)

NNlib.relu6(x::AbstractProbeNumber) = clamp(x, 0, 6)

# TODO: Implement for other scalar types as well?
function NNlib.rrelu(x::BroadcastProbe, lo=1 / 8, hi=1 / 3)
    x = convert(BroadcastProbe{float(eltype(x))}, x)
    a = onnx_op("RandomUniformLike", x; attr=(low=Float32(lo), high=Float32(hi)))
    return onnx_op("PRelu", x, a)
end

NNlib.selu(x::AbstractProbeNumber) = act_fun("Selu", x)

NNlib.sigmoid(x::AbstractProbeNumber) = act_fun("Sigmoid", x)

NNlib.sigmoid_fast(x::AbstractProbeNumber) = sigmoid(x)

NNlib.softplus(x::AbstractProbeNumber) = act_fun("Softplus", x)

function NNlib.softshrink(x::AbstractProbeNumber, λ=0.5)
    return act_fun("Shrink", x; bias=Float32(λ), lambd=Float32(λ))
end

NNlib.softsign(x::AbstractProbeNumber) = act_fun("Softsign", x)

function NNlib.swish(x::AbstractProbeNumber)
    # TODO: Check opset version >= 24.
    # return act_fun("Swish", x)
    return x * sigmoid(x)
end

NNlib.hardswish(x::AbstractProbeNumber) = act_fun("HardSwish", x)

NNlib.tanh_fast(x::AbstractProbeNumber) = tanh(x)

# Implemented in NNlib.
# NNlib.tanhshrink(x::AbstractProbeNumber) = x - tanh(x)

function NNlib.trelu(x::AbstractProbeNumber, theta=1)
    return act_fun("ThresholdedRelu", x; alpha=Float32(theta))
end
