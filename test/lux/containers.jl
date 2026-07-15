@testset "Containers" begin
    @info "Containers"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = BranchLayer(Dense(8 => 4, relu), Dense(8 => 4, tanh_fast); fusion=(+))

    x = rand(rng, Float32, 8, 3)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = PairwiseFusion(vcat, Dense(8 => 4, relu), Dense(12 => 4, relu))

    x = rand(rng, Float32, 8, 3)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = Parallel(+, Dense(8 => 4, relu), Dense(12 => 4, relu))

    x = rand(rng, Float32, 8, 3)
    y = rand(rng, Float32, 12, 3)

    y, y_onnx = test_model(rng, model, (x, y))
    @test y_onnx ≈ y

    model = SkipConnection(Dense(4 => 4, relu), +)

    x = rand(rng, Float32, 4, 3)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = RepeatedLayer(Dense(4 => 4, relu); repeats=Val(2))

    x = rand(rng, Float32, 4, 3)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = AlternatePrecision(
        Float16,
        Dense(
            4 => 4,
            relu;
            init_weight=(rng, out, in) -> randn(rng, Float16, out, in),
            init_bias=(rng, out) -> randn(rng, Float16, out),
        ),
    )

    x = rand(rng, Float32, 4, 3)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y atol=1e-2
end
