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

    model = RotaryPositionalEmbedding(8; max_sequence_length=64)

    x = rand(rng, Float32, 8, 4, 16, 3)
    y, y_onnx = test_model(rng, model, x; opset_version=23)
    @test y_onnx ≈ y

    model = RotaryPositionalEmbedding(8; max_sequence_length=64)

    x = rand(rng, Float32, 8, 4, 16, 3)
    input_pos = rand(rng, Int32.(1:64), 16)
    y, y_onnx = test_model(rng, model, (x, input_pos); opset_version=23)
    @test y_onnx ≈ y

    model = SinusoidalPositionalEmbedding(8)

    x = rand(rng, Float32, 8, 4, 16, 3)
    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    cos_cache, sin_cache = compute_rotary_embedding_params(8, 64; base=10000)
    f1(x) = apply_rotary_embedding(x, cos_cache, sin_cache; seq_dim=4)
    x = rand(rng, Float32, 8, 3, 4, 16)

    y, y_onnx = test_function(f1, x; opset_version=23)
    @test y_onnx ≈ y

    f2(x, y) = apply_rotary_embedding(x, y, cos_cache, sin_cache; seq_dim=2)
    x = rand(rng, Float32, 8, 16, 4, 3)
    y = rand(rng, 1:64, 16)

    y, y_onnx = test_function(f2, x, y; opset_version=23)
    @test y_onnx ≈ y
end
