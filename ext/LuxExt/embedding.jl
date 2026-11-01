# These Embedding implementations are only needed to avoid asserts.
function (e::Embedding)(x::NTuple{N,<:ProbeVector{T}}, ps, st::NamedTuple) where {N,T}
    return Lux.NNlib.gather(ps.weight, x...), st
end
function (e::Embedding)(x::NTuple{N,<:ProbeArray{T}}, ps, st::NamedTuple) where {N,T}
    y, stₙ = e(vec.(x), ps, st)
    return reshape(y, :, size(first(x))...), stₙ
end

function (spe::SinusoidalPositionalEmbedding)(x::ProbeArray, ps, st::NamedTuple)
    return ONNXExport.with_prefix("SinusoidalPositionalEmbedding") do
        y = ONNXExport.unsqueeze(match_eltype(spe, ps, st, x), 1)
        y = y .* probe(st.sigmas, "sigmas")
        z = vcat(sin.(y), cos.(y)) .* probe(spe.scale, "scale")
        return z, st
    end
end

function Lux.apply_rotary_embedding(
    x::ProbeArray{T,4},
    input_positions::AbstractVector{<:Integer},
    cos_cache::AbstractMatrix,
    sin_cache::AbstractMatrix;
    seq_dim::Integer,
) where {T}
    if !ONNXExport.has_minimum_opset("", 23)
        error("ONNX export of apply_rotary_embedding requires opset version 23 or later.")
    end

    # ONNX expects the input size (head_size, sequence_length, num_heads, batch_size).
    if seq_dim == 3
        perm = (1, 3, 2, 4)
    elseif seq_dim == 4
        perm = (1, 4, 2, 3)
    else
        perm = (1, 2, 3, 4)
    end

    return ONNXExport.with_prefix("apply_rotary_embedding") do
        if seq_dim != 2
            @warn "ONNX export of apply_rotary_embedding with seq_dim != 2 requires " *
                "additional permutations of the dimensions. Avoid this by setting " *
                "seq_dim = 2."
            x = permutedims(x, perm)
        end

        # We convert to probe before repeat in case size(x, 4) is a symbolic dimension.
        position_ids = probe(Int64.(input_positions) .- one(Int64), "position_ids")
        position_ids = repeat(position_ids; outer=(1, size(x, 4)))

        cos_cache = probe(cos_cache, "cos_cache")
        sin_cache = probe(sin_cache, "sin_cache")

        y = onnx_op("RotaryEmbedding", raw_size(x), x, cos_cache, sin_cache, position_ids)

        if seq_dim != 2
            y = permutedims(y, invperm(perm))
        end

        return y
    end
end
