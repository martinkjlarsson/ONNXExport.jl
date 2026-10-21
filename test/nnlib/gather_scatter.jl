@testset "Gather" begin
    @info "NNlib - Gather"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    D = Int16[3, 4, 1, 2]
    f1(A, B, C) = NNlib.gather(A, B, C, D)
    A = reshape(1:(3 * 4 * 5), 3, 4, 5)
    B = Int32[1, 2, 3, 2]
    C = [2, 4, 2, 2]

    y, y_onnx = test_function(f1, A, B, C)
    @test y_onnx ≈ y

    C = [1, 2, 3]
    f2(A, B) = NNlib.gather(A, B) + 100*NNlib.gather(A, C)
    A = [1, 20, 300, 4000]
    B = [2, 4, 2]

    y, y_onnx = test_function(f2, A, B)
    @test y_onnx ≈ y

    f3(A, B) = NNlib.gather(A, B)
    A = [1 2 3; 4 5 6]
    B = Int32[1, 3, 1, 3, 1]

    y, y_onnx = test_function(f3, A, B)
    @test y_onnx ≈ y

    f4(A, B) = NNlib.gather(A, B)
    A = [1 2 3; 4 5 6]
    B = [1 2; 2 1]

    y, y_onnx = test_function(f4, A, B)
    @test y_onnx ≈ y
end

@testset "Scatter" begin
    @info "NNlib - Scatter"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(A) = NNlib.scatter(min, 1:8, A) + 100 * NNlib.scatter(max, 1:8, A)
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

    f4(A, B) = NNlib.scatter(*, A, B; init=0.5, dstsize=(2, 4))
    A = Float64[1 10; 100 1000]
    B = Int32[3, 2]

    y, y_onnx = test_function(f4, A, B)
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
