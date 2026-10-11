@testset "Attention" begin
    @info "Lux - Attention"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Chain(
        MultiHeadAttention(4; nheads=2, attention_dropout_probability=0.1),
        first,
        MultiHeadAttention(4; nheads=2, is_causal=true),
    )

    # 3D arrays.
    q = randn(rng, Float32, 4, 2, 3)
    k = randn(rng, Float32, 4, 8, 3)
    v = randn(rng, Float32, 4, 8, 3)
    mask = rand(rng, Bool, 8, 2, 1, 1)
    mask[1, :, 1, 1] .= true

    y, y_onnx = test_model(rng, model, (q, k, v, mask); test=true)
    @test y_onnx[1] ≈ y[1]
    @test y_onnx[2] ≈ y[2]

    # Attention operator requires version 23.
    y, y_onnx = test_model(rng, model, (q, k, v, mask); test=true, opset_version=23)
    @test y_onnx[1] ≈ y[1]
    @test y_onnx[2] ≈ y[2]

    y, y_onnx = test_model(rng, model, (q, k, v, mask); test=false, opset_version=23)
    # We cannot compare random outputs.

    # 4D arrays.
    q = randn(rng, Float32, 4, 2, 3, 2)
    k = randn(rng, Float32, 4, 8, 3, 2)
    v = randn(rng, Float32, 4, 8, 3, 2)

    y, y_onnx = test_model(rng, model, (q, k, v, mask); test=true)
    @test y_onnx[1] ≈ y[1]
    @test y_onnx[2] ≈ y[2]

    # Attention operator requires version 23.
    y, y_onnx = test_model(rng, model, (q, k, v, mask); test=true, opset_version=23)
    @test y_onnx[1] ≈ y[1]
    @test y_onnx[2] ≈ y[2]
end
