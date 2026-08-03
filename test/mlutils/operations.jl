@testset "Operations" begin
    @info "MLUtils - Operations"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(x, y) = flatten(x) .+ unsqueeze(y; dims=1)
    x = rand(rng, Float32, 2, 3, 4)
    y = rand(rng, Float32, 4)

    y, y_onnx = test_function(f1, x, y)
    @test y_onnx ≈ y

    f2(x) = normalise(x; ϵ=1e-9) + normalise(x; dims=(1, 2), ϵ=1e-9)
    x = rand(rng, Float32, 2, 3, 4)

    y, y_onnx = test_function(f2, x)
    @test y_onnx ≈ y

    f3(x) = rescale(x) + rescale(x; dims=(1, 2))
    x = rand(rng, Float32, 2, 3, 4)

    y, y_onnx = test_function(f3, x)
    @test y_onnx ≈ y

    f4(x) = vcat(topk(x, 3)...)
    x = rand(rng, Float32, 7, 8, 9)

    y, y_onnx = test_function(f4, x)
    @test y_onnx ≈ y

    f5(x) = .+(chunk(x, 3; dims=1)...) + 10 * .+(chunk(x; size=3, dims=1)...)
    x = rand(rng, Float32, 7, 4)

    y, y_onnx = test_function(f5, x)
    @test y_onnx ≈ y
end
