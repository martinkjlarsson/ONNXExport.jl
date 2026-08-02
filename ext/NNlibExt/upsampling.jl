function NNlib.upsample_nearest(x::ProbeArray, scales::NTuple{S,<:Integer}) where {S}
    full_scales = ntuple(i -> i <= S ? Int(scales[i]) : 1, ndims(x))
    new_dims = ONNXExport.mul_dim.(raw_size(x), full_scales)

    roi = ProbeArray{Float32}("")
    scales_probe = probe(collect(Float32, reverse(full_scales)), "scales")
    attr=(mode="nearest",)

    return onnx_op("Resize", new_dims, x, roi, scales_probe; attr=attr)
end

function NNlib.upsample_nearest(x::ProbeArray; size::NTuple{S,<:Integer}) where {S}
    new_dims = ntuple(i -> i <= S ? Int(size[i]) : raw_size(x, i), ndims(x))

    roi = ProbeArray{Float32}("")
    scales = ProbeArray{Float32}("")
    sizes_probe = probe(collect(Int64, size), "sizes")
    attr=(axes=ndims(x) .- (1:S), mode="nearest")

    return onnx_op("Resize", new_dims, x, roi, scales, sizes_probe; attr=attr)
end

function NNlib.upsample_linear(
    x::ProbeArray, scales::NTuple{S,Real}; align_corners::Bool=true
) where {S}
    if !all(isa.(scales, Integer)) && !align_corners
        @warn "ONNX export of upsampling with non-integer scale factors and " *
            "align_corners=false may not match NNlib output exactly."
    end

    full_scales = ntuple(i -> i <= S ? Float32(scales[i]) : 1, ndims(x))
    real_dims = ONNXExport.mul_dim.(raw_size(x), full_scales)
    new_dims = map(d -> d isa Symbol ? d : floor(Int, d), real_dims)

    roi = ProbeArray{Float32}("")
    scales_probe = probe(collect(Float32, reverse(full_scales)), "scales")
    ctm = align_corners ? "align_corners" : "half_pixel"
    attr=(coordinate_transformation_mode=ctm, mode="linear")

    return onnx_op("Resize", new_dims, x, roi, scales_probe; attr=attr)
end

function NNlib.upsample_linear(
    x::ProbeArray; size::Union{Integer,NTuple{<:Any,<:Integer}}, align_corners::Bool=true
)
    if size isa Integer
        size = (size,)
    end
    S = length(size)

    new_dims = ntuple(i -> i <= S ? Int(size[i]) : raw_size(x, i), ndims(x))

    roi = ProbeArray{Float32}("")
    scales = ProbeArray{Float32}("")
    sizes_probe = probe(collect(Int64, size), "sizes")
    ctm = align_corners ? "align_corners" : "half_pixel"
    attr=(axes=ndims(x) .- (1:S), coordinate_transformation_mode=ctm, mode="linear")

    return onnx_op("Resize", new_dims, x, roi, scales, sizes_probe; attr=attr)
end

NNlib.pixel_shuffle(x::ProbeArray, r::Integer) = pixel_shuffle(x, Int(r))
function NNlib.pixel_shuffle(x::ProbeArray, r::Int)
    if ndims(x) != 4
        error("ONNX export of pixel_shuffle currently only supports 4D arrays.")
    end
    new_dims = (
        ONNXExport.mul_dim(raw_size(x, 1), r),
        ONNXExport.mul_dim(raw_size(x, 2), r),
        ONNXExport.div_dim(raw_size(x, 3), r*r),
        raw_size(x, 4),
    )
    return onnx_op("DepthToSpace", new_dims, x; attr=(blocksize=r, mode="CRD"))
end
