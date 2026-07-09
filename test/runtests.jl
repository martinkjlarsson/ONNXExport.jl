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
end
