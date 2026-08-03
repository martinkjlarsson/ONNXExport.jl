@testset "Activation" begin
    @info "NNlib - Activation"
    act_funcs = [
        celu,
        elu,
        gelu,
        gelu_tanh,
        gelu_sigmoid,
        gelu_erf,
        hardsigmoid,
        hardtanh,
        leakyrelu,
        lisht,
        logcosh,
        logsigmoid,
        mish,
        relu,
        relu6,
        # rrelu, # Tested below.
        selu,
        sigmoid,
        softplus,
        softshrink,
        softsign,
        swish,
        hardswish,
        tanhshrink,
        trelu,
    ]
    f1(x) = hcat(map(f -> f(x), act_funcs)...)
    x = range(-10.0f0, 10.0f0, 21)

    y, y_onnx = test_function(f1, x)
    @test y_onnx ≈ y

    f2(x) = rrelu.(x)
    x = range(-10.0f0, 10.0f0, 21)

    y, y_onnx = test_function(f2, x)
    # @test y_onnx ≈ y # We cannot compare random outputs.
end
