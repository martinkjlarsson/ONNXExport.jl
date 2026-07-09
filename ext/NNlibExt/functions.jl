function NNlib.softmax(x::ProbeArray; dims::Integer=1)
    return onnx_op("Softmax", x; attr=(axis=ndims(x) - dims,))
end
function NNlib.logsoftmax(x::ProbeArray; dims::Integer=1)
    return onnx_op("LogSoftmax", x; attr=(axis=ndims(x) - dims,))
end

NNlib.logsumexp(x::ProbeArray; dims=:) = ONNXExport._reduce("ReduceLogSumExp", x, dims)

function NNlib.glu(x::ProbeArray, dim::Integer=1)
    new_dims = ntuple(i -> i == dim ? div_dim(raw_size(x, i), 2) : raw_size(x, i), ndims(x))
    a = value_info(eltype(x), new_dims, "a")
    b = value_info(eltype(x), new_dims, "b")
    onnx_op("Split", x, (a, b); attr=(axis=ndims(x) - dim, num_outputs=2))

    return a .* sigmoid.(b)
end
