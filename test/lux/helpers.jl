@testset "Helpers" begin
    @info "Helpers"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Chain(FlattenLayer(), NoOpLayer())

    x = rand(rng, Float32, 8, 6, 4, 2)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = FlattenLayer(2)

    x = rand(rng, Float32, 8, 6, 4, 2)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = Maxout(Dense(8 => 4, relu), Dense(8 => 4, tanh_fast))

    x = rand(rng, Float32, 8, 3)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = ReshapeLayer((4, 2))

    x = rand(rng, Float32, 8, 3)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = Chain(
        SelectDim(1, 3), WrappedFunction(Base.Fix1(broadcast, relu)), ReverseSequence()
    )

    x = rand(rng, Float32, 8, 3)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y
end
