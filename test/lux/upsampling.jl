@testset "Upsampling" begin
    @info "Upsampling"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = PixelShuffle(3)
    x = rand(rng, Float32, 2, 3, 18, 2)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = Upsample(2)
    x = rand(rng, Float32, 2, 3, 4)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = Upsample(:bilinear; size=(4, 9))
    x = rand(rng, Float32, 2, 3, 4, 1)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y
end
