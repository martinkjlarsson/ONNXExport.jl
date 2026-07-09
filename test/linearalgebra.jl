@testset "LinearAlgebra" begin
    @info "LinearAlgebra"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(A, k) = tril(A) + 42 * triu(A) + 4 * tril(A, 2) + 8 * triu(A, 2) + 16 * tril(A, k)
    A = rand(rng, Float32, 5, 5)
    k = 1

    y, y_onnx = test_function(f1, A, k)
    @test y_onnx ≈ y

    f2(A, n) =
        det(A) * UniformScaling(1.0f0)(5) +
        UniformScaling(2.0f0)(n) +
        UniformScaling(8.0f0)(n) +
        dot(A, A) * A
    A = rand(rng, Float32, 5, 5)
    n = 5

    y, y_onnx = test_function(f2, A, n)
    @test y_onnx ≈ y

    f3(A) =
        norm(A, 2) +
        norm(A, 1) +
        norm(A, Inf) +
        norm(A, 0) +
        norm(A, -Inf) +
        opnorm(A, 1) +
        opnorm(A, Inf) +
        sum(normalize(vec(A))) +
        sum(normalize(vec(A), 1))
    A = Float32[0 1 2; 3 4 5]

    y, y_onnx = test_function(f3, A)
    @test y_onnx ≈ y

    f4(A, B, a, b) = kron(A, B) + kron(a, b) .* ones(Float32, 1, 6)
    A = Float32[0 1 2; 3 4 5]
    B = Float32[6 7; 8 9; 10 11; 12 13]
    a = Float32[1, 2]
    b = Float32[3, 4, 5, 6]

    y, y_onnx = test_function(f4, A, B, a, b)
    @test y_onnx ≈ y
end
