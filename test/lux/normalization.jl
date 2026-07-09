@testset "Normalization" begin
    @info "Normalization"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Chain(
        LayerNorm(
            (8, 1),
            relu;
            dims=1,
            affine=true,
            init_bias=glorot_normal, # To avoid trivial initialization.
            init_scale=glorot_normal,
        ),
        LayerNorm((8, 4); dims=1:2, affine=false),
    )
    A = rand(rng, Float32, 8, 4, 3)

    y, y_onnx = test_model(rng, model, A)
    @test y_onnx ≈ y

    # This tests the ONNX Runtime bug https://github.com/microsoft/onnxruntime/issues/27455.
    model = Chain(
        SkipConnection(Dense(4 => 4, relu), +),
        LayerNorm(
            (4, 1);
            dims=1,
            affine=true,
            init_bias=glorot_normal, # To avoid trivial initialization.
            init_scale=glorot_normal,
        ),
    )
    A = rand(rng, Float32, 4, 2, 3)

    y, y_onnx = test_model(rng, model, A)
    @test y_onnx ≈ y
end
