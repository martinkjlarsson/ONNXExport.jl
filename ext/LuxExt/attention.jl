# The Attention ONNX operator is not supported before opset 23,
# so we use more primitive operators instead.

# TODO: Provide options/config so we can check opset version and use Attention if available.

# TODO: The @overload macro caused issues with incremental compilation.
# @overload LuxLib.scaled_dot_product_attention _attention ProbeArray AbstractArray 3
LuxLib.scaled_dot_product_attention(q::ProbeArray,k::ProbeArray,v::ProbeArray; kwargs...) = _attention(q,k,v; kwargs...)
LuxLib.scaled_dot_product_attention(q::ProbeArray,k::ProbeArray,v::AbstractArray; kwargs...) = _attention(q,k,v; kwargs...)
LuxLib.scaled_dot_product_attention(q::ProbeArray,k::AbstractArray,v::ProbeArray; kwargs...) = _attention(q,k,v; kwargs...)
LuxLib.scaled_dot_product_attention(q::ProbeArray,k::AbstractArray,v::AbstractArray; kwargs...) = _attention(q,k,v; kwargs...)
LuxLib.scaled_dot_product_attention(q::AbstractArray,k::ProbeArray,v::ProbeArray; kwargs...) = _attention(q,k,v; kwargs...)
LuxLib.scaled_dot_product_attention(q::AbstractArray,k::ProbeArray,v::AbstractArray; kwargs...) = _attention(q,k,v; kwargs...)
LuxLib.scaled_dot_product_attention(q::AbstractArray,k::AbstractArray,v::ProbeArray; kwargs...) = _attention(q,k,v; kwargs...)

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
    @assert isnothing(mask) "mask is not supported for ONNX export"
    @assert is_causal != true "(casual) mask is not supported for ONNX export"

    scale = scale === nothing ? sqrt(TQ(size(q, head_dim))) : TQ(scale)
    scale = inv(scale)

    @assert ndims(q) == ndims(k) == ndims(v) "Expected q, k, and v to have same number of dimensions."
    q = permutedims(q, (1, 3, 2, 4:ndims(q)...))
    kt = permutedims(k, (3, 1, 2, 4:ndims(q)...))
    v = permutedims(v, (1, 3, 2, 4:ndims(q)...))

    attn_scores = fdrop(softmax(matmul_onnx(kt, q .* scale)))

    x = matmul_onnx(v, attn_scores)
    x = permutedims(x, (1, 3, 2, 4:ndims(x)...))

    return x, attn_scores
end
