module MicrofloatsExt

using ONNXExport.ONNXHelper, Microfloats

ONNXHelper.tensor_type(::Type{Float8_E4M3FN}) = var"TensorProto.DataType".FLOAT8E4M3FN
ONNXHelper.julia_type(::Val{var"TensorProto.DataType".FLOAT8E4M3FN}) = Float8_E4M3FN
function ONNXHelper.tensor_to_field(array::AbstractArray{Float8_E4M3FN})
    return reinterpret(UInt8, vec(array))
end

function ONNXHelper.to_array_typed(
    ::Val{var"TensorProto.DataType".FLOAT8E4M3FN}, tensor::TensorProto
)
    vec_data = [reinterpret(Float8_E4M3FN, UInt8(x)) for x in tensor.int32_data]
    return reshape(vec_data, reverse(tensor.dims)...)
end

ONNXHelper.tensor_type(::Type{Float8_E5M2}) = var"TensorProto.DataType".FLOAT8E5M2
ONNXHelper.julia_type(::Val{var"TensorProto.DataType".FLOAT8E5M2}) = Float8_E5M2
function ONNXHelper.tensor_to_field(array::AbstractArray{Float8_E5M2})
    return reinterpret(UInt8, vec(array))
end

function ONNXHelper.to_array_typed(
    ::Val{var"TensorProto.DataType".FLOAT8E5M2}, tensor::TensorProto
)
    vec_data = [reinterpret(Float8_E5M2, UInt8(x)) for x in tensor.int32_data]
    return reshape(vec_data, reverse(tensor.dims)...)
end

ONNXHelper.tensor_type(::Type{Float8_E8M0FNU}) = var"TensorProto.DataType".FLOAT8E8M0
ONNXHelper.julia_type(::Val{var"TensorProto.DataType".FLOAT8E8M0}) = Float8_E8M0FNU
function ONNXHelper.tensor_to_field(array::AbstractArray{Float8_E8M0FNU})
    return reinterpret(UInt8, vec(array))
end

function ONNXHelper.to_array_typed(
    ::Val{var"TensorProto.DataType".FLOAT8E8M0}, tensor::TensorProto
)
    vec_data = [reinterpret(Float8_E8M0FNU, UInt8(x)) for x in tensor.int32_data]
    return reshape(vec_data, reverse(tensor.dims)...)
end

end
