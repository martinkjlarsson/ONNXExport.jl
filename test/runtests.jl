using BFloat16s
using LinearAlgebra
using Logging
using Lux
using MLUtils
using Microfloats
using NNlib
using ONNXExport
using ONNXRunTime
using Random
using SpecialFunctions
using Statistics
using Test

include("utils.jl")

@testset "ONNXExport.jl" begin
    @testset "Base" begin
        include("base.jl")
        include("linearalgebra.jl")
    end
    @testset "Lux" begin
        include("lux/attention.jl")
        include("lux/containers.jl")
        include("lux/conv.jl")
        include("lux/dropout.jl")
        include("lux/helpers.jl")
        include("lux/linear.jl")
        include("lux/normalization.jl")
        include("lux/pooling.jl")
        include("lux/upsampling.jl")
    end
    @testset "MLUtils" begin
        include("mlutils/array_constructors.jl")
        include("mlutils/operations.jl")
    end
    @testset "NNlib" begin
        include("nnlib/activation.jl")
        include("nnlib/functions.jl")
        include("nnlib/padding.jl")
        include("nnlib/upsampling.jl")
    end
    @testset "SpecialFunctions" begin
        include("special_functions/erf.jl")
    end
end
