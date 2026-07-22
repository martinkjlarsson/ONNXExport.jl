module ONNXHelper

using ProtoBuf
using SparseArrays

include("onnx3_pb.jl")
include("attribute.jl")
include("tensor.jl")
include("sparse_tensor.jl")
include("value_type_shape.jl")
include("node.jl")
include("function.jl")
include("graph.jl")
include("model.jl")
include("io.jl")
include("rename.jl")
include("optimize.jl")

export load_model, save_model
export tensor_type, julia_type, to_array, to_sparse_array, attr_type
export rename_value, rename_value!
export Dimension,
    TensorValueInfoProto,
    SparseTensorValueInfoProto,
    TensorTypeProto,
    SequenceTypeProto,
    MapTypeProto,
    OptionalTypeProto,
    SparseTensorTypeProto

end
