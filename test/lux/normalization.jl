@testset "Normalization" begin
    @info "Normalization"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Chain(
        BatchNorm(
            4,
            sigmoid;
            affine=true,
            momentum=0.15f0,
            init_bias=glorot_normal, # To avoid trivial initialization.
            init_scale=glorot_normal,
        ),
        BatchNorm(4; affine=false),
    )

    A = rand(rng, Float32, 8, 4, 3)

    y, y_onnx = test_model(rng, model, A; test=true)
    @test y_onnx ≈ y

    # TODO: Uncomment when ONNXRunTime issue is resolved.
    # y, y_onnx = test_model(rng, model, A)
    # @test y_onnx ≈ y

    model = Chain(
        GroupNorm(
            4,
            2,
            sigmoid;
            affine=true,
            init_bias=glorot_normal, # To avoid trivial initialization.
            init_scale=glorot_normal,
        ),
        GroupNorm(4, 2, sigmoid; affine=false),
    )

    A = rand(rng, Float32, 2, 4, 3)

    y, y_onnx = test_model(rng, model, A)
    @test y_onnx ≈ y

    model = Chain(
        InstanceNorm(
            4,
            sigmoid;
            affine=true,
            init_bias=glorot_normal, # To avoid trivial initialization.
            init_scale=glorot_normal,
        ),
        InstanceNorm(4, sigmoid; affine=false),
    )

    A = rand(rng, Float32, 2, 4, 3)

    y, y_onnx = test_model(rng, model, A)
    @test y_onnx ≈ y

    model = Chain(
        LayerNorm(
            (8, 1),
            sigmoid;
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
        SkipConnection(Dense(4 => 4, sigmoid), +),
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

    model = WeightNorm(Dense(2 => 3, relu), (:weight,))

    A = rand(rng, Float32, 2, 4, 3)

    y, y_onnx = test_model(rng, model, A)
    @test y_onnx ≈ y

    model = Chain(RMSNorm((2, 4), affine=true), RMSNorm((2, 1); affine=false))

    A = rand(rng, Float32, 2, 4, 3)

    y, y_onnx = test_model(rng, model, A)
    @test y_onnx ≈ y
end
