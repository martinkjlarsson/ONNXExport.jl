NNlib.batched_mul(A::ProbeArray, B::AbstractArray) = matmul_onnx(A, B)
NNlib.batched_mul(A::AbstractArray, B::ProbeArray) = matmul_onnx(A, B)
NNlib.batched_mul(A::ProbeArray, B::ProbeArray) = matmul_onnx(A, B)

NNlib.batched_transpose(A::ProbeArray{T,3}) where {T} = permutedims(A, (2, 1, 3))
NNlib.batched_adjoint(A::ProbeArray{T,3}) where {T<:Real} = permutedims(A, (2, 1, 3))
function NNlib.batched_adjoint(::ProbeArray{T,3}) where {T}
    return error("ONNX does not support complex conjugation.")
end

NNlib.batched_vec(A::ProbeArray, B::AbstractArray) = _batched_vec(A, B)
NNlib.batched_vec(A::AbstractArray, B::ProbeArray) = _batched_vec(A, B)
NNlib.batched_vec(A::ProbeArray, B::ProbeArray) = _batched_vec(A, B)

NNlib.batched_vec(A::ProbeArray{T,3} where {T}, B::AbstractMatrix) = _batched_vec(A, B)
NNlib.batched_vec(A::AbstractArray{T,3} where {T}, B::ProbeMatrix) = _batched_vec(A, B)
NNlib.batched_vec(A::ProbeArray{T,3} where {T}, B::ProbeMatrix) = _batched_vec(A, B)

NNlib.batched_vec(A::ProbeArray{T,3} where {T}, b::AbstractVector) = _batched_vec(A, b)
NNlib.batched_vec(A::AbstractArray{T,3} where {T}, b::ProbeVector) = _batched_vec(A, b)
NNlib.batched_vec(A::ProbeArray{T,3} where {T}, b::ProbeVector) = _batched_vec(A, b)

function _batched_vec(A, B)
    B_reshaped = ONNXExport.unsqueeze(B, 2)
    C = batched_mul(A, B_reshaped)
    return dropdims(C; dims=2)
end

function _batched_vec(A, b::AbstractVector)
    b_reshaped = ONNXExport.unsqueeze(b, (2, 3))
    C = batched_mul(A, b_reshaped)
    return dropdims(C; dims=2)
end
