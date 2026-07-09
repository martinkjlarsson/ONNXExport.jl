@testset "Dense" begin
    @info "Dense"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Chain(
        Dense(16 => 12, relu), Dense(12 => 8, leakyrelu; use_bias=false), Dense(8 => 4)
    )

    x = rand(rng, Float32, 16)
    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    x = rand(rng, Float32, 16, 4)
    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    x = rand(rng, Float32, 16, 4, 3)
    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y
end

@testset "Bilinear" begin
    @info "Bilinear"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Bilinear((8, 12) => 6, relu)

    x1 = rand(rng, Float32, 8, 4)
    x2 = rand(rng, Float32, 12, 4)
    y, y_onnx = test_model(rng, model, (x1, x2))
    @test y_onnx ≈ y
end

@testset "Scale" begin
    @info "Scale"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Chain(Scale(8, relu), Scale(8, leakyrelu; use_bias=false), Scale(8))

    x = rand(rng, Float32, 8)
    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    x = rand(rng, Float32, 8, 4)
    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    x = rand(rng, Float32, 8, 4, 3)
    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y
end
