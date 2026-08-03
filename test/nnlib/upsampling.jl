@testset "Upsampling" begin
    @info "NNlib - Upsampling"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(x) = upsample_nearest(x, (2, 3))
    x = rand(rng, Float32, 4, 3, 2)

    y, y_onnx = test_function(f1, x)
    @test y_onnx ≈ y

    f2(x) = upsample_nearest(x; size=(8, 9))
    x = rand(rng, Float32, 4, 3, 2)

    y, y_onnx = test_function(f2, x)
    @test y_onnx ≈ y

    f3(x) = upsample_linear(x, 2.2; align_corners=true)
    x = rand(rng, Float32, 4, 3, 2)

    y, y_onnx = test_function(f3, x)
    @test y_onnx ≈ y

    f4(x) = upsample_bilinear(x, (2.2, 2.9); align_corners=true)
    x = rand(rng, Float32, 4, 3, 2, 1)

    y, y_onnx = test_function(f4, x)
    @test y_onnx ≈ y

    # Note: We need to pick scales carefully for the outputs to match.
    f5(x) = upsample_linear(x, 2.5; align_corners=false)
    x = rand(rng, Float32, 4, 3, 2)

    y, y_onnx = test_function(f5, x)
    @test y_onnx ≈ y

    # Note: We need to pick scales carefully for the outputs to match.
    f6(x) = upsample_bilinear(x, (2.5, 10/3); align_corners=false)
    x = rand(rng, Float32, 4, 3, 2, 1)

    y, y_onnx = test_function(f6, x)
    @test y_onnx ≈ y

    f7(x) = upsample_linear(x; size=10, align_corners=true)
    x = rand(rng, Float32, 4, 3, 2)

    y, y_onnx = test_function(f7, x)
    @test y_onnx ≈ y

    f8(x) = upsample_bilinear(x; size=(6, 10), align_corners=true)
    x = rand(rng, Float32, 4, 3, 2, 1)

    y, y_onnx = test_function(f8, x)
    @test y_onnx ≈ y

    f9(x) = pixel_shuffle(x, 3)
    x = rand(rng, Float32, 2, 3, 18, 2)

    y, y_onnx = test_function(f9, x)
    @test y_onnx ≈ y
end
