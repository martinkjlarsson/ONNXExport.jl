@testset "Batched" begin
    @info "NNlib - Batched"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(A, B) = NNlib.batched_mul(A, B)
    A = rand(rng, Float32, 3, 4, 5)
    B = rand(rng, Float32, 4, 2, 5)

    y, y_onnx = test_function(f1, A, B)
    @test y_onnx ≈ y

    f2(A) = NNlib.batched_transpose(A) + NNlib.batched_adjoint(A)
    A = rand(rng, Float32, 3, 4, 5)

    y, y_onnx = test_function(f2, A)
    @test y_onnx ≈ y

    f3(A, B) = NNlib.batched_vec(A, B)
    A = rand(rng, Float32, 3, 4, 5)
    B = rand(rng, Float32, 4, 5)

    y, y_onnx = test_function(f3, A, B)
    @test y_onnx ≈ y

    f4(A, B) = NNlib.batched_vec(A, B)
    A = rand(rng, Float32, 3, 4, 5)
    B = rand(rng, Float32, 4)

    y, y_onnx = test_function(f4, A, B)
    @test y_onnx ≈ y
end
