@testset "Padding" begin
    @info "Padding"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    for padding in [pad_reflect, pad_symmetric, pad_circular, pad_repeat]
        f1(x) = padding(x, (2, 3))
        x = rand(rng, Float32, 4, 3, 2, 1)

        y, y_onnx = test_function(f1, x)
        @test y_onnx ≈ y

        f2(x) = padding(x, 2)
        x = rand(rng, Float32, 4, 3, 2, 1)

        y, y_onnx = test_function(f2, x)
        @test y_onnx ≈ y
    end

    f3(x) = pad_constant(x, 2, 1.5f0) + pad_zeros(x, 2)
    x = rand(rng, Float32, 4, 3, 2, 1)

    y, y_onnx = test_function(f3, x)
    @test y_onnx ≈ y

    f4(x) = pad_constant(x, (1, 2); dims=(1, 3))
    x = rand(rng, Float32, 4, 3, 2, 1)

    y, y_onnx = test_function(f4, x)
    @test y_onnx ≈ y
end
