@testset "Pooling" begin
    @info "Lux - Pooling"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    for model in
        [AdaptiveLPPool((10, 6); p=1), AdaptiveMaxPool((10, 6)), AdaptiveMeanPool((10, 6))]
        x = rand(rng, Float32, 12, 9, 4, 2)

        y, y_onnx = test_model(rng, model, x)
        @test y_onnx ≈ y
    end

    for model in [GlobalLPPool(; p=1), GlobalMaxPool(), GlobalMeanPool()]
        x = rand(rng, Float32, 12, 9, 4, 2)

        y, y_onnx = test_model(rng, model, x)
        @test y_onnx ≈ y
    end

    for model in [
        LPPool((3, 5); stride=(1, 2), pad=(1, 2, 3, 4), dilation=(2, 1), p=1),
        MaxPool((3, 5); stride=(1, 2), pad=(1, 2, 3, 4), dilation=(2, 1)),
        MeanPool((3, 5); stride=(1, 2), pad=(1, 2, 3, 4), dilation=(2, 1)),
    ]
        x = rand(rng, Float32, 12, 9, 4, 2)

        y, y_onnx = test_model(rng, model, x)
        @test y_onnx ≈ y
    end
end
