"Cast an integer array to Int32s or Int64s."
cast_index(x::AbstractArray{<:Integer}) = eltype(x) == Int32 ? x : Int.(x)

NNlib.gather(src::AbstractArray, idx::ProbeArray) = _gather(src, idx)
NNlib.gather(src::ProbeArray, idx::AbstractArray) = _gather(src, idx)
NNlib.gather(src::ProbeArray, idx::ProbeArray) = _gather(src, idx)

function _gather(
    src::AbstractArray{Tsrc,Nsrc}, idx::AbstractArray{Tidx,Nidx}
) where {Tsrc,Nsrc,Nidx,Tidx}
    if !(Tidx <: Integer)
        # Support for constant cartesian indices can be added using the GatherND operator.
        error("ONNX export of gather only supports integer indices.")
    end
    M = 1 # Integers have length 1.

    src = probe(src)
    if isprobe(idx) # TODO: Should raw_size be implemented for AbstractArray?
        dstsize = (raw_size(src)[1:(Nsrc - M)]..., raw_size(idx)...)
    else
        dstsize = (raw_size(src)[1:(Nsrc - M)]..., size(idx)...)
    end

    if Nsrc == Nidx == 1
        idx = cast_index(idx)
        idx = idx .- one(eltype(idx))
        return onnx_op("GatherElements", dstsize, src, probe(idx))
    else
        idx = Int64.(idx)
        idx = idx .- one(eltype(idx))
        idx = ONNXExport.unsqueeze(idx, 1)
        return onnx_op("GatherND", dstsize, src, probe(idx))
    end
end

function NNlib.gather(
    src::ProbeArray,
    I::AbstractVector{<:Integer},
    J::AbstractVector{<:Integer},
    Ks::AbstractVector{<:Integer}...,
)
    return _gather_ijk(src, I, J, Ks...)
end
function NNlib.gather(
    src::AbstractArray,
    I::ProbeVector{<:Integer},
    J::AbstractVector{<:Integer},
    Ks::AbstractVector{<:Integer}...,
)
    return _gather_ijk(src, I, J, Ks...)
end
function NNlib.gather(
    src::ProbeArray,
    I::ProbeVector{<:Integer},
    J::AbstractVector{<:Integer},
    Ks::AbstractVector{<:Integer}...,
)
    return _gather_ijk(src, I, J, Ks...)
end
# TODO: More overloads using the @overload macro.

function _gather_ijk(src, I, J, Ks...)
    M = 2 + length(Ks)

    I = Int64.(I)
    J = Int64.(J)
    Ks = map(x -> Int64.(x), Ks)

    src = probe(src)
    indices = vcat(transpose.(reverse(Ks))..., J', I')
    indices = indices .- one(eltype(indices))
    indices = probe(indices)

    dstsize = (raw_size(src)[1:(end - M)]..., raw_size(indices, 2))
    return onnx_op("GatherND", dstsize, src, indices)
end

function NNlib.scatter(
    op::OP, src::AbstractArray{Tsrc,Nsrc}, idx::ProbeArray{Tidx,Nidx}; kwargs...
) where {Tsrc,Tidx,Nsrc,Nidx,OP}
    return _scatter(op, src, idx; kwargs...)
end
function NNlib.scatter(
    op::OP, src::ProbeArray{Tsrc,Nsrc}, idx::AbstractArray{Tidx,Nidx}; kwargs...
) where {Tsrc,Tidx,Nsrc,Nidx,OP}
    return _scatter(op, src, idx; kwargs...)
end
function NNlib.scatter(
    op::OP, src::ProbeArray{Tsrc,Nsrc}, idx::ProbeArray{Tidx,Nidx}; kwargs...
) where {Tsrc,Tidx,Nsrc,Nidx,OP}
    return _scatter(op, src, idx; kwargs...)
end

function _scatter(
    op::OP,
    src::AbstractArray{Tsrc,Nsrc},
    idx::AbstractArray{Tidx,Nidx};
    init=nothing,
    dstsize=nothing,
) where {Tsrc,Tidx,Nsrc,Nidx,OP}
    if !(Tidx <: Integer)
        # Support for constant cartesian indices can be added using the ScatterND operator.
        error("ONNX export of scatter with non-integer indices is not supported.")
    end

    dims = Nsrc - Nidx
    dstsz =
        isnothing(dstsize) ? (size(src)[1:dims]..., NNlib.maximum_dims(idx)...) : dstsize
    xinit = isnothing(init) ? NNlib.scatter_empty(op, Tsrc) : init

    dst = fill(Tsrc(xinit), dstsz)

    if op === +
        reduction = "add"
        required_version = 16
    elseif op === *
        reduction = "mul"
        required_version = 16
    elseif op === max
        reduction = "max"
        required_version = 18
    elseif op === min
        reduction = "min"
        required_version = 18
    else
        throw(ArgumentError("ONNX export of scatter with op=$op is not supported."))
    end
    if !ONNXExport.has_minimum_opset("", required_version)
        throw(
            ArgumentError(
                "ONNX export of scatter with op=$op requires opset version " *
                "$required_version or later.",
            ),
        )
    end

    # ONNX input naming.
    data = probe(dst)
    indices = idx
    updates = src

    if dims == 0
        indices = cast_index(indices)
        indices = indices .- one(eltype(indices))

        # data is a vector and the other inputs must have the same number of dimensions.
        indices = vec(indices)
        updates = vec(updates)
        return onnx_op(
            "ScatterElements",
            raw_size(data),
            data,
            probe(indices),
            probe(updates);
            attr=(reduction=reduction,),
        )
    else
        indices = Int64.(indices)
        indices = indices .- one(eltype(indices))
        indices = ONNXExport.unsqueeze(indices, 1)
        return onnx_op(
            "ScatterND",
            raw_size(data),
            data,
            probe(indices),
            probe(updates);
            attr=(reduction=reduction,),
        )
    end
end
