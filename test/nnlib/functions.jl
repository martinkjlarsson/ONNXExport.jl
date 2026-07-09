@testset "Functions" begin
    @info "Functions"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(x) = softmax(x) + logsoftmax(x)
    x = rand(rng, Float32, 6, 4, 2)

    y, y_onnx = test_function(f1, x)
    @test y_onnx ≈ y

    f2(x) = glu(logsumexp(x; dims=2), 1)
    x = rand(rng, Float32, 6, 4, 2)

    y, y_onnx = test_function(f2, x)
    @test y_onnx ≈ y
end
