module LuxExt

using LinearAlgebra, Lux, ONNXExport, Random, Static, Statistics

is_training(::Union{False,Val{false}}) = false
is_training(::Union{True,Val{true}}) = true
is_training(::Nothing) = error("training is set to Nothing, which is not supported")

include("attention.jl")
include("conv.jl")
include("dropout.jl")
include("helpers.jl")
include("linear.jl")
include("luxlib.jl")
include("normalization.jl")

end
