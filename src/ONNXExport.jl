module ONNXExport

using LinearAlgebra
using Random
using SciMLPublic
using Statistics

if VERSION < v"1.11"
    using ScopedValues
else
    using Base.ScopedValues
end

include("ONNXHelper/src/ONNXHelper.jl")
using .ONNXHelper

include("macros.jl")
include("namespace.jl")
include("graph.jl")
include("probe.jl")
include("probearray.jl")
include("probenumber.jl")
include("export.jl")
include("broadcasting.jl")
include("math.jl")
include("indexing.jl")
include("arraymath.jl")
include("linearalgebra.jl")
include("statistics.jl")
include("reduce.jl")
include("operators.jl")
include("controlflow.jl")
include("array.jl")
include("random.jl")

@public save, trace
export NullProbe,
    ProbeArray,
    ProbeMatrix,
    ProbeVector,
    ProbeScalar,
    ProbeNumber,
    AbstractProbeNumber,
    BroadcastProbe,
    ProbeRNG,
    ProbeInteger,
    ProbeIntegers
export raw_size, isprobe, probe
export matmul_onnx, onnx_op, value_info

end
