@testset "Convolution" begin
    @info "NNlib - Convolution"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    weight = rand(Float32, 3, 5, 2, 2)
    f1(x) = conv(
        x,
        weight;
        stride=(1, 2),
        dilation=(2, 1),
        pad=(1, 2, 3, 4),
        groups=2,
        flipped=true,
    )

    x = rand(rng, Float32, 12, 9, 4, 1)

    y, y_onnx = test_function(f1, x)
    @test y_onnx ≈ y

    weight = rand(Float32, 3, 5, 2, 2)
    f2(x) = conv(
        x,
        weight;
        stride=(1, 2),
        dilation=(2, 1),
        pad=(1, 2, 3, 4),
        groups=2,
        flipped=false,
    )

    x = rand(rng, Float32, 12, 9, 4, 1)

    y, y_onnx = test_function(f2, x)
    @test y_onnx ≈ y

    weight = rand(Float32, 3, 5, 2, 4)
    f3(x) = depthwiseconv(
        x, weight; stride=(1, 2), dilation=(2, 1), pad=(1, 2, 3, 4), flipped=false
    )

    x = rand(rng, Float32, 12, 9, 4, 1)

    y, y_onnx = test_function(f3, x)
    @test y_onnx ≈ y
end
