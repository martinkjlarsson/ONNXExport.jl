function Statistics.mean(A::ProbeArray; dims=:)
    return _reduce("ReduceMean", A, dims)
end
function Statistics.mean(f, A::ProbeArray; dims=:)
    return mean(f.(A); dims=dims)
end

# Statistics.jl defines mean(itr) = mean(identity, itr).
Statistics.mean(f, itr::AbstractVector{<:ProbeArray}) = _mean(f.(itr))
Statistics.mean(f, itr::ProbeTuple) = _mean(f.(itr))
function _mean(itr)
    isempty(itr) && error("Cannot reduce over empty collection")
    itr = probe(itr)
    T = float(promote_type(eltype.(itr)...))
    itr = map(v -> convert(ProbeArray{T}, v), itr)
    return onnx_op("Mean", itr...)
end
