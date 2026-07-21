@testset "ErrorFunctions" begin
    @info "ErrorFunctions"

    f1(A, b) = erf.(A) .+ erf(b)
    A = rand(Float32, 2, 3)
    b = rand(Float32)

    y, y_onnx = test_function(f1, A, b)
    @test y_onnx ≈ y
end
