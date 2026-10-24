function NNlib.imrotate(
    arr::ProbeArray{T,4}, θ; method=:bilinear, rotation_center=nothing, pixel_perfect=false
) where {T}
    if !ONNXExport.has_minimum_opset("", 20)
        error("ONNX imrotate requires opset version 20 or later.")
    end

    if !(T <: AbstractFloat)
        throw(ArgumentError("ONNX export of imrotate requires floating-point input."))
    end

    if method == :bilinear
        mode = "linear"
    elseif method == :nearest
        mode = "nearest"
    else
        throw(ArgumentError("$method is not a valid interpolation method."))
    end

    if !pixel_perfect
        @warn "ONNX export of imrotate might not match NNlib implementation pixel perfect."
    end

    # TODO: Add support for symbolic dimensions.
    input_size = raw_size(arr)
    size1, size2, channels, batch = input_size
    if !(input_size isa Dims)
        throw(
            ArgumentError(
                "ONNX export of imrotate requires statically known input dimensions."
            ),
        )
    end

    if isnothing(rotation_center)
        rotation_center = (size1 ÷ 2 + 1, size2 ÷ 2 + 1)
    elseif length(rotation_center) != 2
        throw(ArgumentError("rotation_center must contain two coordinates."))
    end

    center1, center2 = T.(rotation_center)
    sinθ, cosθ = sincos(T(θ))

    center_x = 2 * (center1 - T(0.5)) / size1 - 1
    center_y = 2 * (center2 - T(0.5)) / size2 - 1
    tx = center_x - cosθ * center_x + sinθ * (size2 / size1) * center_y
    ty = center_y - sinθ * (size1 / size2) * center_x - cosθ * center_y
    rot_mat = T[
        cosθ sinθ * size1 / size2
        -sinθ * size2 / size1 cosθ
        tx ty
    ]
    theta = repeat(rot_mat; outer=(1, 1, batch))
    theta_probe = probe(theta, "theta")
    size_probe = probe(Int64[batch, channels, size2, size1], "size")

    grid = onnx_op(
        "AffineGrid",
        T,
        (2, size1, size2, batch),
        theta_probe,
        size_probe;
        attr=(align_corners=0,),
    )

    if pixel_perfect && method == :nearest
        # ONNX rounds exact half-pixel coordinates to even, while NNlib rounds ties away
        # from zero. A tiny positive bias makes nearest sampling agree in tie cases.
        sample_grid = grid .+ eps(T) * 8
    else
        sample_grid = grid
    end

    sampled = onnx_op(
        "GridSample",
        input_size,
        arr,
        sample_grid;
        attr=(align_corners=0, mode=mode, padding_mode="zeros"),
    )

    if pixel_perfect && method == :bilinear
        # NNlib returns zero unless both interpolation neighbors are in bounds, resulting in
        # aliasing against the black background. ONNX does not do this and have anti-aliased
        # edges. This mess ensures the ONNX export is pixel perfect.
        grid_x = reshape(grid[1, :, :, :], (size1, size2, 1, batch))
        grid_y = reshape(grid[2, :, :, :], (size1, size2, 1, batch))
        tol = eps(T)
        valid_x = (grid_x .>= T(1 / size1 - (1+tol))) .& (grid_x .< T(1 - 1 / size1))
        valid_y = (grid_y .>= T(1 / size2 - (1+tol))) .& (grid_y .< T(1 - 1 / size2))
        return ifelse.(valid_x .& valid_y, sampled, zero(T))
    end

    return sampled
end
