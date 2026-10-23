@testset "Sampling" begin
    @info "NNlib - Sampling"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    x = reshape(collect(1.0:4.0), (2, 2, 1, 1))
    grid = Array{Float64}(undef, 2, 3, 2, 1)
    grid[:, 1, 1, 1] .= (-3, -1)
    grid[:, 2, 1, 1] .= (0, -1)
    grid[:, 3, 1, 1] .= (1, -1)
    grid[:, 1, 2, 1] .= (-1, 1)
    grid[:, 2, 2, 1] .= (0, 1)
    grid[:, 3, 2, 1] .= (3, 1)

    f1(x, grid) = grid_sample(x, grid)

    y, y_onnx = test_function(f1, x, grid)
    @test y_onnx ≈ y

    f2(x, grid) = grid_sample(x, grid; padding_mode=:border)

    y, y_onnx = test_function(f2, x, grid)
    @test y_onnx ≈ y

    f3(x, grid) = grid_sample(x, grid; padding_mode=:zeros)
    x = rand(rng, Float32, 2, 3, 4, 2, 2)
    grid = 1 .- 2 .* rand(rng, Float32, 3, 3, 1, 1, 2)

    y, y_onnx = test_function(f3, x, grid)
    @test y_onnx ≈ y
end
