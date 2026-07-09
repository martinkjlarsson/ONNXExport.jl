using LinearAlgebra
using Logging
using Lux
using NNlib
using ONNXExport
using ONNXRunTime
using Random
using Statistics
using Test

include("utils.jl")

@testset "ONNXExport.jl" begin
    @testset "Base" begin
        include("base.jl")
        include("linearalgebra.jl")
    end
    @testset "Lux" begin
        include("lux/linear.jl")
        include("lux/normalization.jl")
        include("lux/attention.jl")
        include("lux/conv.jl")
        include("lux/pooling.jl")
    end
    @testset "NNlib" begin
        include("nnlib/activation.jl")
        include("nnlib/functions.jl")
    end
end
