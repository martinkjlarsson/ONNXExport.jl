NNlib.batched_mul(A::ProbeArray, B::AbstractArray) = matmul_onnx(A, B)
NNlib.batched_mul(A::AbstractArray, B::ProbeArray) = matmul_onnx(A, B)
NNlib.batched_mul(A::ProbeArray, B::ProbeArray) = matmul_onnx(A, B)
