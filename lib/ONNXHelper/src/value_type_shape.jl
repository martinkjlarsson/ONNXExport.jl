const Dimension = var"TensorShapeProto.Dimension"

function ValueInfoProto(
    name::String, type::TypeProto; doc_string="", metadata_props=StringStringEntryProto[]
)
    return ValueInfoProto(name, type, doc_string, metadata_props)
end

# TODO: Would ValueInfoProtoTensor improve code completion?
function TensorValueInfoProto(
    name::String,
    type::Type{T},
    shape;
    doc_string="",
    metadata_props=StringStringEntryProto[],
    denotation="",
) where {T}
    type_proto = TensorTypeProto(type, shape; denotation=denotation)
    return ValueInfoProto(name, type_proto, doc_string, metadata_props)
end

function SparseTensorValueInfoProto(
    name::String,
    type::Type{T},
    shape;
    doc_string="",
    metadata_props=StringStringEntryProto[],
    denotation="",
) where {T}
    type_proto = SparseTensorTypeProto(type, shape; denotation=denotation)
    return ValueInfoProto(name, type_proto, doc_string, metadata_props)
end

# TODO: Would TypeProtoTensor improve code completion?
function TensorTypeProto(::Type{T}, shape::TensorShapeProto; denotation="") where {T}
    value = var"TypeProto.Tensor"(Int32(tensor_type(T)), shape)
    return TypeProto(OneOf(:tensor_type, value), denotation)
end
function TensorTypeProto(::Type{T}, shape; denotation="") where {T}
    sp = TensorShapeProto(shape)
    return TensorTypeProto(T, sp; denotation=denotation)
end

function SequenceTypeProto(elem_type::TypeProto; denotation="")
    value = var"TypeProto.Sequence"(elem_type)
    return TypeProto(OneOf(:sequence_type, value), denotation)
end

function MapTypeProto(::Type{T}, value_type::TypeProto; denotation="") where {T}
    value = var"TypeProto.Map"(Int32(tensor_type(T)), value_type)
    return TypeProto(OneOf(:map_type, value), denotation)
end

function OptionalTypeProto(elem_type::TypeProto; denotation="")
    value = var"TypeProto.Optional"(elem_type)
    return TypeProto(OneOf(:optional_type, value), denotation)
end

function SparseTypeProto(::Type{T}, shape::TensorShapeProto; denotation="") where {T}
    value = var"TypeProto.SparseTensor"(Int32(tensor_type(T)), shape)
    return TypeProto(OneOf(:sparse_tensor_type, value), denotation)
end
function SparseTensorTypeProto(::Type{T}, shape; denotation="") where {T}
    sp = TensorShapeProto(shape)
    return SparseTypeProto(T, sp; denotation=denotation)
end

function TensorShapeProto(shape::Tuple)
    return TensorShapeProto([d isa Dimension ? d : Dimension(d) for d in shape])
end
function TensorShapeProto(shape::Union{Integer,AbstractString,Symbol,Nothing})
    return TensorShapeProto(Dimension(shape))
end
function TensorShapeProto(shape::Dimension)
    return TensorShapeProto([shape])
end

function Dimension(value::Integer, denotation="")
    return Dimension(OneOf{Int64}(:dim_value, value), denotation)
end
function Dimension(value::AbstractString, denotation="")
    return Dimension(OneOf{String}(:dim_param, value), denotation)
end
function Dimension(value::Symbol, denotation="")
    return Dimension(OneOf{String}(:dim_param, string(value)), denotation)
end
function Dimension(::Nothing)
    return Dimension(nothing, "")
end
