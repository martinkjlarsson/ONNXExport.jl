@testset "Rotation" begin
    @info "NNlib - Rotation"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    # We ensure the imrotate export is pixel perfect here for testing purposes. In practice,
    # the default pixel_perfect=false is likely close enough.
    function pixel_perfect_imrotate(x, theta; kwargs...)
        if isprobe(x)
            return NNlib.imrotate(x, theta; pixel_perfect=true, kwargs...)
        else
            return NNlib.imrotate(x, theta; kwargs...)
        end
    end

    f1(x) = pixel_perfect_imrotate(x, deg2rad(30); method=:nearest, rotation_center=(2, 3))
    x = rand(rng, Float64, 5, 5, 2, 2)

    y, y_onnx = test_function(f1, x)
    @test y_onnx ≈ y

    f2(x) = pixel_perfect_imrotate(x, deg2rad(30))
    x = rand(rng, Float32, 4, 7, 2, 2)

    y, y_onnx = test_function(f2, x)
    @test y_onnx ≈ y

    f3(x) = pixel_perfect_imrotate(x, 0.123456; rotation_center=(1, 7))
    x = rand(rng, Float32, 63, 64, 3, 2)

    y, y_onnx = test_function(f3, x)
    @test y_onnx ≈ y
end
