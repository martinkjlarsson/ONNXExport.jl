module NNlibExt

using NNlib, ONNXExport

include("activation.jl")
include("functions.jl")
include("batched.jl")
include("pooling.jl")

end
