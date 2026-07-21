module SpecialFunctionsExt

using SpecialFunctions, ONNXExport

function SpecialFunctions.erf(x::AbstractProbeNumber)
    return onnx_op("Erf", float(eltype(x))(x))
end
function SpecialFunctions.erf(x::AbstractProbeNumber{<:AbstractFloat})
    return onnx_op("Erf", x)
end

end
