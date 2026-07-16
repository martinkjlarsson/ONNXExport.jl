@testset "Types" begin
    @info "Types"

    f1(A, B) = A * B
    A = rand(1:10, 3, 3)
    B = rand(Float32, 3, 3)

    y, y_onnx = test_function(f1, A, B)
    @test y_onnx ≈ y
end

@testset "Array" begin
    @info "Array"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(A, B, C) = [A B; C]
    A = rand(rng, Float32, 2, 3)
    B = rand(rng, Float32, 2)
    C = rand(rng, Float32, 1, 4)

    y, y_onnx = test_function(f1, A, B, C)
    @test y_onnx ≈ y

    f2(A) = reshape(permutedims(A, (4, 1, 3, 2)), 5, 3, count(A .<= 4), :)
    A = reshape(Float32.(1:(2 * 3 * 4 * 5)), 2, 3, 4, 5)

    y, y_onnx = test_function(f2, A)
    @test y_onnx ≈ y

    f3(A) = repeat(A; inner=(1, 2, 3, 4), outer=(3, 1, 2, 4))
    A = rand(rng, Float32, 2, 3, 4, 5)

    y, y_onnx = test_function(f3, A)
    @test y_onnx ≈ y

    C = rand(rng, Float32, 3, 6)
    f4(A, B, D) = cat(A, B, C, D; dims=2)
    A = rand(rng, Float32, 3, 2)
    B = rand(rng, Int, 3, 4)
    D = rand(rng, Float32, 3)

    y, y_onnx = test_function(f4, A, B, D)
    @test y_onnx ≈ y

    f5(A) = cumsum(A; dims=2) + 100 * cumprod(A; dims=2)
    A = [1 2 3 4 5; 2 2 2 2 2]

    y, y_onnx = test_function(f5, A)
    @test y_onnx ≈ y

    f6(A) = cumsum(A) + 100 * cumprod(A)
    A = [1, 2, 3, 4, 5, 6, 7, 8]

    y, y_onnx = test_function(f6, A)
    @test y_onnx ≈ y

    f6a(A) =
        accumulate(+, A; init=7.0) +
        (10 * accumulate(+, A; dims=1, init=7.0f0)) +
        (100 * accumulate(+, A; dims=2, init=7)) +
        1000 * accumulate(+, A; dims=3, init=7)
    A = [1 2 3 4 5; 2 2 2 2 2]

    y, y_onnx = test_function(f6a, A)
    @test y_onnx ≈ y

    f7(v, d1, d2) = fill(v, 3) .+ fill(v, 3, d1, d2) .+ fill(2.3, 3, d1, d2)
    v = 3.4
    d1 = 3
    d2 = 0x2

    y, y_onnx = test_function(f7, v, d1, d2)
    @test y_onnx ≈ y

    f8(v) = hcat(
        partialsort(v, 1:4),
        partialsort(v, 1:4; rev=true),
        partialsort(v, 4:-1:1),
        partialsort(v, 2:5),
    )
    v = Float32[1, 3, 5, 7, 2, 4, 6, 8]

    y, y_onnx = test_function(f8, v)
    @test y_onnx ≈ y

    f9(A, B) = dropdims(A; dims=(2, 5)) + ONNXExport.unsqueeze(B, 2)
    A = rand(Float32, 2, 1, 1, 3, 1)
    B = rand(Float32, 2, 3)

    y, y_onnx = test_function(f9, A, B)
    @test y_onnx ≈ y
end

@testset "Indexing" begin
    @info "Indexing"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    indsA = [true false true; false false true]
    indsB = [true, true, false, false, true]
    f1(A, B) = A[indsA] + B[indsB]
    A = rand(rng, Float32, 2, 3)
    B = rand(rng, Float32, 5)

    y, y_onnx = test_function(f1, A, B)
    @test y_onnx ≈ y

    f2(A, B, indsA, indsB) = A[indsA] + B[indsB]
    A = rand(rng, Float32, 2, 3)
    B = rand(rng, Float32, 5)
    indsA = [true false true; false false true]
    indsB = [true, true, false, false, true]

    y, y_onnx = test_function(f2, A, B, indsA, indsB)
    @test y_onnx ≈ y

    f3(A) = A[[true, true, false], :, BitVector([1, 0, 0, 1, 1])]
    A = rand(rng, Float32, 3, 4, 5)

    y, y_onnx = test_function(f3, A)
    @test y_onnx ≈ y

    f4(A) = A[[1, 3, 2, 2], 3, 5:-2:2, [0x2, 0x4, 0x3]]
    A = rand(rng, Float32, 3, 4, 5, 6)

    y, y_onnx = test_function(f4, A)
    @test y_onnx ≈ y

    f5(A, I1, I2, I3, I4) = A[I1, I2, I3, I4]
    A = rand(rng, Float32, 3, 4, 5, 6)
    I1 = Int32[1, 3, 2, 2]
    I2 = 3
    I3 = 5:-2:2
    I4 = [true, true, false, false, true, false]

    y, y_onnx = test_function(f5, A, I1, I2, I3, I4)
    @test y_onnx ≈ y

    A = rand(Float32, 2, 3)
    f6(inds) = A[:, inds]
    inds = [1, 3, 2]

    y, y_onnx = test_function(f6, inds)
    @test y_onnx ≈ y

    f7(A, ind) = (B=A[ind]; return B[size(B, 1)])
    A = rand(Float32, 2, 3)
    ind = [true false true; false false true]

    y, y_onnx = test_function(f7, A, ind)
    @test y_onnx ≈ y

    f8(A) = A[4:6] + 8 * A[3:-1:1]
    A = Float32[1, 2, 3, 4, 5, 6]

    y, y_onnx = test_function(f8, A)
    @test y_onnx ≈ y

    f9(A, i) = selectdim(A, 2, 2) + selectdim([7 8 9; 10 11 12], 2, i)
    A = Float32[1 2 3; 4 5 6]
    i = 3

    y, y_onnx = test_function(f9, A, i)
    @test y_onnx ≈ y
end

@testset "Math" begin
    @info "Math"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(A, B) = matmul_onnx(A, B) # Batched matmul.
    A = rand(rng, Float32, 3, 4, 2, 1)
    B = rand(rng, Float32, 4, 5, 2, 3)

    y, y_onnx = test_function(f1, A, B)
    @test y_onnx ≈ y

    f2(a, B, C) = a * B * C + 2 * B * C
    a = 2.0f0
    B = rand(rng, Float32, 2, 3)
    C = rand(rng, Float32, 3, 4)

    y, y_onnx = test_function(f2, a, B, C)
    @test y_onnx ≈ y

    f3(A) = inv(A)
    A = rand(rng, Float32, 4, 4)

    y, y_onnx = test_function(f3, A)
    @test y_onnx ≈ y

    f4(A, B) = A + B + [1 2 3; 4 5 6]
    A = rand(rng, Float32, 2, 3)
    B = rand(rng, Float32, 2, 3)

    y, y_onnx = test_function(f4, A, B)
    @test y_onnx ≈ y

    f5(a, b) = -a + mod(a, 2) + 8 * max(a, b) + 16 * min(a, b)
    a = 3.14159f0
    b = 2.71828f0

    y, y_onnx = test_function(f5, a, b)
    @test y_onnx ≈ y

    f6(a, b) = (a << 2) | (a >>> 2) & b
    a = rand(UInt)
    b = rand(UInt)

    y, y_onnx = test_function(f6, a, b)
    @test y_onnx ≈ y

    sf1(x) = sin(x) + abs(x) * x
    sf2(x) = abs(sin(x))
    sf3(x) = x + isfinite(x)
    sf4(x) = (cos(x) * sin(x))^15

    f7(x) = (sf1 ∘ sf2 ∘ sf3)(sf4.(x))
    x = rand(Float32)

    y, y_onnx = test_function(f7, x)
    @test y_onnx ≈ y
end

@testset "Broadcasting" begin
    @info "Broadcasting"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(x) = sin.(x) .+ abs.(x) .* x
    f2(x) = abs.(sin.(x))
    f3(x) = x + isfinite.(x)
    f4(x) = (cos(x) * sin(x))^15

    f5(x) = (f1 ∘ f2 ∘ f3)(f4.(x))
    x = rand(rng, Float32, 3, 4)

    y, y_onnx = test_function(f5, x)
    @test y_onnx ≈ y

    f6(A) = clamp.(A, 0.3, 0.6) + round.(100 * A) + floor.(Int, 100 * A)
    A = rand(rng, Float32, 3, 4)

    y, y_onnx = test_function(f6, A)
    @test y_onnx ≈ y

    f7(A, B, C) = ifelse.(A, B, C) + ifelse.(A, B, Float32[1 2 3; 4 5 6])
    A = rand(rng, Bool, 1, 3)
    B = rand(rng, Float32, 2)
    C = rand(rng, Float32, 2, 3)

    y, y_onnx = test_function(f7, A, B, C)
    @test y_onnx ≈ y
end

@testset "Reduce" begin
    @info "Reduce"

    rng = Random.default_rng()
    Random.seed!(rng, 0)

    f1(A) =
        sum(A) .+ sum(A; dims=1) .+ sum(abs, A; dims=2) .+ sum(abs2, A; dims=(1, 2)) .+
        count(>(0.5), A; dims=1)
    A = rand(rng, 3, 4)

    y, y_onnx = test_function(f1, A)
    @test y_onnx ≈ y

    f2(A) = prod(A) .+ prod(A; dims=1)
    A = rand(rng, 3, 4)

    y, y_onnx = test_function(f2, A)
    @test y_onnx ≈ y

    f3(A) = maximum(A) .+ maximum(A; dims=1) .+ minimum(A) .+ minimum(A; dims=2)
    A = rand(rng, 3, 4)

    y, y_onnx = test_function(f3, A)
    @test y_onnx ≈ y

    f4(A) = any(A; dims=1) .& .!all(A; dims=1)
    A = [false true true true; false false true true; false false false true]

    y, y_onnx = test_function(f4, A)
    @test y_onnx ≈ y

    f5(A) = mean(A) .+ mean(A; dims=1) .+ mean(x -> x^2, (2 * A, 10 * A))
    A = rand(rng, Float32, 3, 3)

    y, y_onnx = test_function(f5, A)
    @test y_onnx ≈ y

    f6(v) = argmax(v) + argmin(v)
    v = rand(rng, Float32, 10)

    y, y_onnx = test_function(f6, v)
    @test y_onnx ≈ y

    f7(A) = unique(A; dims=2) .+ 4 * unique(abs2, A)
    A = [1 1 2; -1 -1 -1]

    y, y_onnx = test_function(f7, A)
    @test y_onnx ≈ y
end
