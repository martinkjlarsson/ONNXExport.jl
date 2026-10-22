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

        position_ids = repeat(Int64.(input_positions) .- one(Int64); outer=(1, size(x, 4)))

        cos_cache = probe(cos_cache, "cos_cache")
        sin_cache = probe(sin_cache, "sin_cache")
        position_ids = probe(position_ids, "position_ids")

        y = onnx_op("RotaryEmbedding", raw_size(x), x, cos_cache, sin_cache, position_ids)

        if seq_dim != 2
            y = permutedims(y, invperm(perm))
        end

        return y
    end
end
