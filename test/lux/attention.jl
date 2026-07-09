@testset "Attention" begin
    @info "Attention"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    model = Chain(
        MultiHeadAttention(4; nheads=2), first, MultiHeadAttention(4; nheads=2), first
    )

    q = randn(Float32, 4, 2, 3)
    k = randn(Float32, 4, 8, 3)
    v = randn(Float32, 4, 8, 3)

    y, y_onnx = test_model(rng, model, (q, k, v))
    @test y_onnx ≈ y
end
