@testset "Convolution" begin
    @info "Convolution"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Conv(
        (3, 5),
        4 => 2,
        relu;
        use_bias=true,
        stride=(1, 2),
        dilation=(2, 1),
        pad=(1, 2, 3, 4),
        groups=2,
        cross_correlation=true,
    )

    x = rand(rng, Float32, 12, 9, 4, 1)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y

    model = Conv(
        (3, 5),
        4 => 2;
        use_bias=false,
        stride=(1, 2),
        dilation=(2, 1),
        pad=(1, 2, 3, 4),
        groups=2,
        cross_correlation=false,
    )

    x = rand(rng, Float32, 12, 9, 4, 1)

    y, y_onnx = test_model(rng, model, x)
    @test y_onnx ≈ y
end
