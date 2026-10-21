@testset "Scatter" begin
    @info "NNlib - Scatter"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(A) = NNlib.scatter(max, 1:8, A)
    A = [1, 2, 3, 4, 1, 2, 3, 4]

    y, y_onnx = test_function(f1, A)
    @test y_onnx ≈ y

    f2(A) = NNlib.scatter(+, [1 2 3 4; 5 6 7 8], A)
    A = [2, 1, 1, 5]

    y, y_onnx = test_function(f2, A)
    @test y_onnx ≈ y

    f3(A) = NNlib.scatter(*, [10, 200, 3000], A; init=10, dstsize=6)
    A = [1, 4, 2]

    y, y_onnx = test_function(f3, A)
    @test y_onnx ≈ y

    f4(A) = NNlib.scatter(*, A, [3, 2]; init=0.5, dstsize=(2, 4))
    A = Float64[1 10; 100 1000]

    y, y_onnx = test_function(f4, A)
    @test y_onnx ≈ y

    f5(A, B) = NNlib.scatter(*, A, B)
    A = rand(rng, 4, 4, 2)
    B = [1, 3]

    y, y_onnx = test_function(f5, A, B)
    @test y_onnx ≈ y

    f6(A, b) = NNlib.scatter(+, A, [1 2; 3 4]; init=b)
    A = [10 30; 20 40]
    b = 2

    y, y_onnx = test_function(f6, A, b)
    @test y_onnx ≈ y
end
