
function NNlib.grid_sample(
    input::AbstractArray{T,N}, grid::ProbeArray; kwargs...
) where {T,N}
    return grid_sample(probe(input), grid; kwargs...)
end

function NNlib.grid_sample(
    input::ProbeArray{T,N}, grid::AbstractArray; kwargs...
) where {T,N}
    return grid_sample(input, probe(grid); kwargs...)
end

function NNlib.grid_sample(
    input::ProbeArray{T,N}, grid::ProbeArray; padding_mode=:zeros
) where {T,N}
    if eltype(grid) != T
        @warn "ONNX export of NNlib.grid_sample with input and grid of different element " *
            "types may not be supported by ONNX Runtime."
    end

    iC = raw_size(input, N-1)
    iN = raw_size(input, N)
    output_size = raw_size(grid)[2:(end - 1)]
    new_dims = (output_size..., iC, iN)

    if padding_mode == :zeros
        padding_mode_str = "zeros"
    elseif padding_mode == :border
        padding_mode_str = "border"
    else
        throw(ArgumentError("$padding_mode is not a valid padding mode."))
    end

    attr = (align_corners=1, padding_mode=padding_mode_str)
    return onnx_op("GridSample", new_dims, input, grid; attr=attr)
end
