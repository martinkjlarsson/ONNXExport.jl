@testset "Audio" begin
    @info "NNlib - Audio"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(n) =
        NNlib.hann_window(n, Float32; periodic=true) +
        100*NNlib.hann_window(n, Float32; periodic=false)
    n = 128

    y, y_onnx = test_function(f1, n)
    @test y_onnx ≈ y

    f2(n) =
        NNlib.hamming_window(n, Float32; periodic=true) +
        100*NNlib.hamming_window(n, Float32; periodic=false)
    n = 128

    y, y_onnx = test_function(f2, n)
    @test y_onnx ≈ y

    f3(n) =
        NNlib.hamming_window(n, Float32; periodic=true, α=0.54347825f0, β=0.45652175f0) +
        100*NNlib.hamming_window(n, Float32; periodic=false, α=0.54347825f0, β=0.45652175f0)
    n = 128

    y, y_onnx = test_function(f3, n)
    @test y_onnx ≈ y

    f4(x) = ONNXExport.stft(x; n_fft=16, hop_length=4, center=true, normalized=false)
    x = rand(rng, Float32, 1, 64, 1)

    y, y_onnx = test_function(f4, x)
    @test y_onnx ≈ y

    f5(x, n) = ONNXExport.stft(
        x; n_fft=n, center=false, normalized=true, window=collect(1.0f0:16.0f0)
    )
    x = rand(rng, Float32, 1, 64, 1)
    n = 16

    y, y_onnx = test_function(f5, x, n)
    @test y_onnx ≈ y

    f6(x) = ONNXExport.stft(x; n_fft=n, hop_length=16, center=true, normalized=false)
    x = rand(rng, Float32, 2, 64, 1)

    y, y_onnx = test_function(f6, x)
    @test y_onnx ≈ y

    f7(x, n) = ONNXExport.stft(
        x;
        n_fft=n,
        center=false,
        normalized=true,
        # window=collect(1.0f0:16.0f0), # TODO: Uncomment when ORT issue is resolved.
    )
    x = rand(rng, Float32, 2, 64, 1)
    n = 16

    y, y_onnx = test_function(f7, x, n)
    @test y_onnx ≈ y
end
