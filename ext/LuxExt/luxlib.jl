function LuxLib.API.batched_matmul(x::AbstractMatrix, y::ProbeArray{yT,3}) where {yT}
    return matmul_onnx(x, y)
end
function LuxLib.API.batched_matmul(x::ProbeMatrix, y::AbstractArray{yT,3}) where {yT}
    return matmul_onnx(x, y)
end
function LuxLib.API.batched_matmul(x::ProbeMatrix, y::ProbeArray{yT,3}) where {yT}
    return matmul_onnx(x, y)
end

function LuxLib.API.batched_matmul(x::AbstractArray{xT,3}, y::ProbeMatrix) where {xT}
    return matmul_onnx(x, y)
end
function LuxLib.API.batched_matmul(x::ProbeArray{xT,3}, y::AbstractMatrix) where {xT}
    return matmul_onnx(x, y)
end
function LuxLib.API.batched_matmul(x::ProbeArray{xT,3}, y::ProbeMatrix) where {xT}
    return matmul_onnx(x, y)
end

function LuxLib.API.batched_matmul(x::AbstractArray, y::ProbeArray; kwargs...)
    @assert isempty(kwargs) "Keyword arguments not yet supported: $kwargs"
    return matmul_onnx(x, y)
end
function LuxLib.API.batched_matmul(x::ProbeArray, y::AbstractArray; kwargs...)
    @assert isempty(kwargs) "Keyword arguments not yet supported: $kwargs"
    return matmul_onnx(x, y)
end
function LuxLib.API.batched_matmul(x::ProbeArray, y::ProbeArray; kwargs...)
    @assert isempty(kwargs) "Keyword arguments not yet supported: $kwargs"
    return matmul_onnx(x, y)
end

function LuxLib.API.bias_activation!!(σ::F, x::ProbeArray, bias::AbstractVector) where {F}
    return σ(x .+ bias)
end
function LuxLib.API.bias_activation!!(σ::F, x::ProbeArray, ::Nothing) where {F}
    return σ(x)
end
