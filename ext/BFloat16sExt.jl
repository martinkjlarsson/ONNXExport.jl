module BFloat16sExt

using ONNXHelper, BFloat16s

ONNXHelper.tensor_type(::Type{BFloat16}) = var"TensorProto.DataType".BFLOAT16
ONNXHelper.julia_type(::Val{var"TensorProto.DataType".BFLOAT16}) = BFloat16
ONNXHelper.tensor_to_field(array::AbstractArray{BFloat16}) = reinterpret(UInt16, vec(array))

function ONNXHelper.to_array_typed(
    ::Val{var"TensorProto.DataType".BFLOAT16}, tensor::TensorProto
)
    vec_data = [reinterpret(BFloat16, UInt16(x)) for x in tensor.int32_data]
    return reshape(vec_data, reverse(tensor.dims)...)
end

end
