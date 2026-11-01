function LuxLib.API.batched_matmul(x::AbstractArray, y::ProbeArray; kwargs...)
    return _batched_matmul(promote(x, y)...; kwargs...)
end
function LuxLib.API.batched_matmul(x::ProbeArray, y::AbstractArray; kwargs...)
    return _batched_matmul(promote(x, y)...; kwargs...)
end
function LuxLib.API.batched_matmul(x::ProbeArray, y::ProbeArray; kwargs...)
    return _batched_matmul(promote(x, y)...; kwargs...)
end
function LuxLib.API.batched_matmul(x::ProbeArray{T}, y::ProbeArray{T}; kwargs...) where {T}
    return _batched_matmul(x, y; kwargs...)
end

function _batched_matmul(
    x::ProbeArray{xT,xN},
    y::ProbeArray{yT,yN};
    lhs_contracting_dim::Int=2,
    rhs_contracting_dim::Int=1,
    lhs_batching_dims::Dims=ntuple(Base.Fix2(+, 2), Val(xN - 2)),
    rhs_batching_dims::Dims=ntuple(Base.Fix2(+, 2), Val(yN - 2)),
) where {xT,yT,xN,yN}
    if (
        lhs_batching_dims != ntuple(Base.Fix2(+, 2), Val(xN - 2)) ||
        rhs_batching_dims != ntuple(Base.Fix2(+, 2), Val(yN - 2)) ||
        lhs_contracting_dim != 2 ||
        rhs_contracting_dim != 1
    )
        lhs_non_contracting_dims = get_non_contracting_dim(
            xN, lhs_contracting_dim, lhs_batching_dims
        )
        rhs_non_contracting_dims = get_non_contracting_dim(
            yN, rhs_contracting_dim, rhs_batching_dims
        )
        x_permuted = Lux.LuxLib.Utils.maybe_permutedims(
            x, (lhs_non_contracting_dims, lhs_contracting_dim, lhs_batching_dims...)
        )
        y_permuted = Lux.LuxLib.Utils.maybe_permutedims(
            y, (rhs_contracting_dim, rhs_non_contracting_dims, rhs_batching_dims...)
        )
    else
        x_permuted, y_permuted = x, y
    end

    return matmul_onnx(x_permuted, y_permuted)
end

function get_non_contracting_dim(
    N::Int, contracting_dim::Int, batching_dims::Dims{M}
) where {M}
    return only(setdiff(1:N, (contracting_dim, batching_dims...)))
end

function LuxLib.API.bias_activation!!(σ::F, x::ProbeArray, bias::AbstractVector) where {F}
    return σ(x .+ bias)
end
function LuxLib.API.bias_activation!!(σ::F, x::ProbeArray, ::Nothing) where {F}
    return σ(x)
end
