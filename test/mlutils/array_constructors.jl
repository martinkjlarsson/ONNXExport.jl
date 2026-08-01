@testset "Array Constructors" begin
    @info "Array Constructors"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(x) = x + fill_like(x, 2) .+ fill_like(x, 3, (3,))
    x = rand(rng, Float32, 3, 4)

    y, y_onnx = test_function(f1, x)
    @test y_onnx ≈ y

    f2(x, n) = f1(repeat(x, 1, n))
    x = rand(rng, Float32, 3)
    n = 4

    y, y_onnx = test_function(f2, x, n)
    @test y_onnx ≈ y

    f3(x) = x + zeros_like(x) + ones_like(x) + falses_like(x) + trues_like(x)
    x = rand(rng, Float32, 3, 4)

    y, y_onnx = test_function(f3, x)
    @test y_onnx ≈ y

    f4(x, n) = f3(repeat(x, 1, n))
    x = rand(rng, Float32, 3)
    n = 4

    y, y_onnx = test_function(f4, x, n)
    @test y_onnx ≈ y

    f5(x) = x + rand_like(x) + randn_like(x)
    x = rand(rng, Float32, 3, 4)

    y, y_onnx = test_function(f5, x)
    # @test y_onnx ≈ y # We cannot compare random outputs.

    f6(x) =
        x +
        rand_like(x, Float64) +
        randn_like(x, Float64) +
        rand_like(x, Float64, (3, 4)) +
        rand_like(x, (3, 4))
    x = rand(rng, Float32, 3, 4)

    y, y_onnx = test_function(f6, x)
    # @test y_onnx ≈ y # We cannot compare random outputs.
end
