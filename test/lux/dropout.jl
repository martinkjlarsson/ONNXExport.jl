@testset "Dropout" begin
    @info "Dropout"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    # Only verify no-op in test mode exports.
    model = AlphaDropout(0.1)

    x = rand(rng, Float32, 16)
    y, y_onnx = test_model(rng, model, x; test=true)
    @test y_onnx ≈ y

    model = Dropout(0.1)

    x = rand(rng, Float32, 16)
    # We cannot compare result as Dropout is random.
    y, y_onnx = test_model(rng, model, x; test=false)
    y, y_onnx = test_model(rng, model, x; test=true)
    @test y_onnx ≈ y

    # Only verify no-op in test mode exports.
    model = VariationalHiddenDropout(0.1)

    x = rand(rng, Float32, 16)
    y, y_onnx = test_model(rng, model, x; test=true)
    @test y_onnx ≈ y
end
