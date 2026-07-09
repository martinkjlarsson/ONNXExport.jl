@testset "Pooling" begin
    @info "Pooling"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    for model in [GlobalLPPool(), GlobalMaxPool(), GlobalMeanPool()]
        x = rand(rng, Float32, 12, 9, 4, 2, 1)

        y, y_onnx = test_model(rng, model, x)
        @test y_onnx ≈ y
    end
end
