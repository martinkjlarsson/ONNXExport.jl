function (mha::MultiHeadAttention)(x::ProbeArray, ps, st::NamedTuple)
    return apply_multiheadattention(mha, ps, st, x)
end
function (mha::MultiHeadAttention)(
    x::ONNXExport.AtLeastOne{ProbeArray,AbstractArray}, ps, st::NamedTuple
)
    return apply_multiheadattention(mha, ps, st, x...)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, qkv)
    return apply_multiheadattention(mha, ps, st, qkv, qkv, qkv, nothing)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, q, kv)
    return apply_multiheadattention(mha, ps, st, q, kv, kv, nothing)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, q, k, v, mask=nothing)
    return ONNXExport.with_prefix("MultiHeadAttention") do
        @assert ndims(q) == ndims(k) == ndims(v)
        N = ndims(q)
        q_dims = size(q)

        q, q_st = mha.q_proj(q, ps.q_proj, st.q_proj)
        k, k_st = mha.k_proj(k, ps.k_proj, st.k_proj)
        v, v_st = mha.v_proj(v, ps.v_proj, st.v_proj)

        dropout = StatefulLuxLayer(
            mha.attention_dropout, ps.attention_dropout, st.attention_dropout
        )

        # The Attention operator does not support dropout.
        has_dropout =
            mha.attention_dropout isa Dropout && is_training(st.attention_dropout.training)

        if ONNXExport.has_minimum_opset("", 23) && !has_dropout
            if N != 3
                q = reshape(q, size(q, 1), size(q, 2), :)
                k = reshape(k, size(k, 1), size(k, 2), :)
                v = reshape(v, size(v, 1), size(v, 2), :)
            end

            attr = (
                qk_matmul_output_mode=3,
                q_num_heads=mha.nheads,
                kv_num_heads=mha.nheads,
                is_causal=mha.is_causal == true, # is_causal has type Union{Bool,Nothing}.
            )

            q = probe(q)
            k = probe(k)
            v = probe(v)
            mask = probe(mask)

            # TODO: Verify dimensions of (q, k, v) and merge dims?

            x = value_info(eltype(v), (raw_size(v, 1), raw_size(q, 2), raw_size(q, 3)), "Y")
            present_key = value_info(eltype(v), (), "present_key")
            present_value = value_info(eltype(v), (), "present_value")
            attn_scores = value_info(
                eltype(v),
                (raw_size(k, 2), raw_size(q, 2), mha.nheads, raw_size(q, 3)),
                "attention",
            )
            onnx_op(
                "Attention",
                (q, k, v, mask),
                (x, present_key, present_value, attn_scores);
                attr=attr,
            )

            if N != 3
                x = reshape(x, size(x, 1), q_dims[2:end]...)
                attn_scores = reshape(
                    attn_scores, (size(attn_scores)[1:3]..., q_dims[3:end]...)
                )
            end
        else
            x, attn_scores = scaled_dot_product_attention(
                reshape(q, size(q, 1) ÷ mha.nheads, mha.nheads, size(q)[2:end]...),
                reshape(k, size(k, 1) ÷ mha.nheads, mha.nheads, size(k)[2:end]...),
                reshape(v, size(v, 1) ÷ mha.nheads, mha.nheads, size(v)[2:end]...);
                head_dim=1,
                token_dim=3,
                fdrop=dropout,
                mask,
                mha.is_causal,
            )
            x = reshape(x, size(x, 1) * mha.nheads, size(x)[3:end]...)
        end

        y, out_st = mha.out_proj(x, ps.out_proj, st.out_proj)

        return (
            (y, attn_scores),
            (;
                q_proj=q_st,
                k_proj=k_st,
                v_proj=v_st,
                attention_dropout=dropout.st,
                out_proj=out_st,
            ),
        )
    end
end

# TODO: The @overload macro caused issues with incremental compilation.
# @overload LuxLib.scaled_dot_product_attention _attention ProbeArray AbstractArray 3
function LuxLib.scaled_dot_product_attention(
    q::ProbeArray, k::ProbeArray, v::ProbeArray; kwargs...
)
    return _attention(q, k, v; kwargs...)
end
function LuxLib.scaled_dot_product_attention(
    q::ProbeArray, k::ProbeArray, v::AbstractArray; kwargs...
)
    return _attention(q, k, v; kwargs...)
end
function LuxLib.scaled_dot_product_attention(
    q::ProbeArray, k::AbstractArray, v::ProbeArray; kwargs...
)
    return _attention(q, k, v; kwargs...)
end
function LuxLib.scaled_dot_product_attention(
    q::ProbeArray, k::AbstractArray, v::AbstractArray; kwargs...
)
    return _attention(q, k, v; kwargs...)
end
function LuxLib.scaled_dot_product_attention(
    q::AbstractArray, k::ProbeArray, v::ProbeArray; kwargs...
)
    return _attention(q, k, v; kwargs...)
end
function LuxLib.scaled_dot_product_attention(
    q::AbstractArray, k::ProbeArray, v::AbstractArray; kwargs...
)
    return _attention(q, k, v; kwargs...)
end
function LuxLib.scaled_dot_product_attention(
    q::AbstractArray, k::AbstractArray, v::ProbeArray; kwargs...
)
    return _attention(q, k, v; kwargs...)
end

function _attention(
    q::AbstractArray{TQ,N},
    k::AbstractArray{TK,N},
    v::AbstractArray{TV,N};
    head_dim::Int=1,
    token_dim::Int=3,
    scale=nothing,
    mask=nothing,
    is_causal::Union{Bool,Nothing}=nothing,
    bias=nothing,
    fdrop::F=identity,
) where {TQ,TK,TV,F,N}
    @assert head_dim == 1 "ONNX export with head_dim = $head_dim is not supported"
    @assert token_dim == 3 "ONNX export with head_dim = $token_dim is not supported"
    @assert isnothing(bias) "bias is not supported for ONNX export"

    # The Attention operator does not support dropout.
    no_dropout =
        fdrop isa StatefulLuxLayer && (
            fdrop.model isa Dropout && !is_training(fdrop.st.training) ||
            fdrop.model isa NoOpLayer
        ) || fdrop == identity

    if ONNXExport.has_minimum_opset("", 23) && no_dropout
        # Note: This branch is only used if scaled_dot_product_attention is called directly.
        #       Normally, the implementation in apply_multiheadattention is used, which
        #       avoids the permutedims below.

        q_dims = size(q)

        if N > 4
            q = reshape(q, size(q, 1), size(q, 2), size(q, 3), :)
            k = reshape(k, size(k, 1), size(k, 2), size(k, 3), :)
            v = reshape(v, size(v, 1), size(v, 2), size(v, 3), :)
        end

        # ONNX requires (head_size, sequence_length, num_heads, batch_size)
        q = permutedims(q, (1, 3, 2, 4))
        k = permutedims(k, (1, 3, 2, 4))
        v = permutedims(v, (1, 3, 2, 4))

        if isnothing(scale)
            attr = (qk_matmul_output_mode=3, is_causal=is_causal == true)
        else
            attr = (qk_matmul_output_mode=3, is_causal=is_causal == true, scale=scale)
        end

        q = probe(q)
        k = probe(k)
        v = probe(v)
        mask = probe(mask)

        # TODO: Verify dimensions of (q, k, v) and merge dims?

        x = value_info(
            eltype(v), (raw_size(v, 1), raw_size(q, 2), raw_size(q, 3), raw_size(q, 4)), "Y"
        )
        present_key = value_info(eltype(v), (), "present_key")
        present_value = value_info(eltype(v), (), "present_value")
        attn_scores = value_info(
            eltype(v),
            (raw_size(k, 2), raw_size(q, 3), raw_size(q, 2), raw_size(q, 4)),
            "attention",
        )
        onnx_op(
            "Attention",
            (q, k, v, mask),
            (x, present_key, present_value, attn_scores);
            attr=attr,
        )

        x = permutedims(x, (1, 3, 2, 4))

        if N > 4
            x = reshape(x, size(x, 1), q_dims[2:end]...)
            attn_scores = reshape(
                attn_scores, (size(attn_scores)[1:3]..., q_dims[4:end]...)
            )
        end

        return x, attn_scores
    else
        if !ONNXExport.has_minimum_opset("", 23)
            @warn "Increasing the opset version to 23 or higher would enable ONNX export " *
                "using the Attention operator, which might be more efficient than the " *
                "current export."
        end

        scale = scale === nothing ? sqrt(TQ(size(q, head_dim))) : TQ(scale)
        scale = inv(scale)

        q = permutedims(q, (1, 3, 2, 4:ndims(q)...))
        kt = permutedims(k, (3, 1, 2, 4:ndims(q)...))
        v = permutedims(v, (1, 3, 2, 4:ndims(q)...))

        logits = matmul_onnx(kt, q .* scale)
        if is_causal == true
            mask = trues(size(kt, 1), size(q, 2))
            triu!(mask)
        end
        if !isnothing(mask)
            logits = ifelse.(mask, logits, typemin(eltype(logits)))
        end
        attn_scores = fdrop(softmax(logits))

        x = matmul_onnx(v, attn_scores)
        x = permutedims(x, (1, 3, 2, 4:ndims(x)...))

        return x, attn_scores
    end
end
