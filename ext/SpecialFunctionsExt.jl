module SpecialFunctionsExt

using SpecialFunctions, ONNXExport

function SpecialFunctions.erf(x::AbstractProbeNumber)
    return onnx_op("Erf", float(x))
end

end
