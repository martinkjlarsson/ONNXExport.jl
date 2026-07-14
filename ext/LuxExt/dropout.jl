dynamic_training(::Union{False,Val{false}}) = false
dynamic_training(::Union{True,Val{true}}) = true
dynamic_training(::Nothing) = error("training is set to Nothing, which is not supported")

function LuxLib.API.dropout(
    rng::AbstractRNG, x::ProbeArray, p::T, training::LuxLib.API.TrainingType, invp::T, dims
) where {T}
    @assert invp == 1 / (1-p) "invp must be equal to 1 / (1-p); got $invp and $(1 / (1-p))"
    @assert dims === (:) "ONNX export of Dropout requires dims to be :"

    ratio = probe(p)
    training_mode = probe(dynamic_training(training))

    output = value_info(eltype(x), raw_size(x), "dropout_output")
    mask = value_info(Bool, raw_size(x), "dropout_mask")
    onnx_op("Dropout", (x, ratio, training_mode), (output, mask))

    return output, mask, rng
end
