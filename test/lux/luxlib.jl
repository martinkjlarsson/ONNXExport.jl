@testset "LuxLib" begin
    @info "Lux - LuxLib"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(A, B) = LuxLib.API.batched_matmul(A, B)
    A = rand(rng, Float32, 3, 4)
    B = rand(rng, Float64, 4, 5, 2)

    y, y_onnx = test_function(f1, A, B)
    @test y_onnx ≈ y

    f2(A, B) = LuxLib.API.batched_matmul(A, B)
    A = rand(rng, Float32, 3, 4, 2)
    B = rand(rng, Float64, 4, 5, 2)

    y, y_onnx = test_function(f2, A, B)
    @test y_onnx ≈ y

    f3(A, B) = LuxLib.API.batched_matmul(A, B; rhs_contracting_dim=2)
    A = rand(rng, Float32, 3, 4, 2)
    B = rand(rng, Float64, 3, 4, 2)

    y, y_onnx = test_function(f3, A, B)
    @test y_onnx ≈ y
end
