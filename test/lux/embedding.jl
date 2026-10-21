@testset "Embedding" begin
    @info "Lux - Embedding"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Embedding(2 => 32)

    x = rand(rng, 1:2, 4, 5)
    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = Embedding((8, 4) => 16)

    x1 = rand(rng, 1:8, 4, 5)
    x2 = rand(rng, 1:4, 4, 5)
    y, y_onnx = test_model(rng, model, (x1, x2))
    @test y_onnx ≈ y
end
