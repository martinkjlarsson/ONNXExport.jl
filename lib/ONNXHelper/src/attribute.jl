attr_type(x) = attr_type(typeof(x))
function attr_type(::Type{T}) where {T}
    throw(ArgumentError("$T does not map to an ONNX attribute type."))
end
attr_type(::Type{Float32}) = var"AttributeProto.AttributeType".FLOAT
attr_type(::Type{<:Integer}) = var"AttributeProto.AttributeType".INT
attr_type(::Type{<:AbstractString}) = var"AttributeProto.AttributeType".STRING
attr_type(::Type{TensorProto}) = var"AttributeProto.AttributeType".TENSOR
attr_type(::Type{GraphProto}) = var"AttributeProto.AttributeType".GRAPH
attr_type(::Type{SparseTensorProto}) = var"AttributeProto.AttributeType".SPARSE_TENSOR
attr_type(::Type{TypeProto}) = var"AttributeProto.AttributeType".TYPE_PROTO
attr_type(::Type{<:AbstractVector{Float32}}) = var"AttributeProto.AttributeType".FLOATS
function attr_type(::Type{<:AbstractVector{<:Integer}})
    return var"AttributeProto.AttributeType".INTS
end
function attr_type(::Type{<:AbstractVector{<:AbstractString}})
    return var"AttributeProto.AttributeType".STRINGS
end
attr_type(::Type{<:AbstractVector{TensorProto}}) = var"AttributeProto.AttributeType".TENSORS
attr_type(::Type{<:AbstractVector{GraphProto}}) = var"AttributeProto.AttributeType".GRAPHS
function attr_type(::Type{<:AbstractVector{SparseTensorProto}})
    return var"AttributeProto.AttributeType".SPARSE_TENSORS
end
function attr_type(::Type{<:AbstractVector{TypeProto}})
    return var"AttributeProto.AttributeType".TYPE_PROTOS
end

attr_conv(v) = v
attr_conv(v::AbstractString) = codeunits(v)
attr_conv(v::AbstractVector{<:AbstractString}) = codeunits.(v)

for (T, F) in (
    (Float32, :f),
    (Integer, :i),
    (AbstractString, :s),
    (TensorProto, :t),
    (GraphProto, :g),
    (SparseTensorProto, :sparse_tensor),
    (TypeProto, :tp),
    (AbstractVector{Float32}, :floats),
    (AbstractVector{<:Integer}, :ints),
    (AbstractVector{<:AbstractString}, :strings),
    (AbstractVector{TensorProto}, :tensors),
    (AbstractVector{GraphProto}, :graphs),
    (AbstractVector{SparseTensorProto}, :sparse_tensors),
    (AbstractVector{TypeProto}, :type_protos),
)
    @eval begin
        function AttributeProto(name::String, value::$T; doc_string="")
            return _attributeproto(;
                name=name,
                doc_string=doc_string,
                var"#type"=attr_type(value),
                $F=attr_conv(value),
            )
        end
    end
end

function AttributeProto(name::String, value::Tuple; kwards...)
    return AttributeProto(name, collect(value); kwards...)
end

function AttributeProto(
    name::String,
    ref_attr_name::String,
    type::var"AttributeProto.AttributeType".T;
    doc_string="",
)
    return _attributeproto(;
        name=name, ref_attr_name=ref_attr_name, doc_string=doc_string, var"#type"=type
    )
end

function AttributeProto(
    name::String, ref_attr_name::String, type::Type{T}; doc_string=""
) where {T}
    return AttributeProto(name, ref_attr_name, attr_type(T); doc_string=doc_string)
end

function _attributeproto(;
    name="",
    ref_attr_name="",
    doc_string="",
    var"#type"=var"AttributeProto.AttributeType".UNDEFINED,
    f=zero(Float32),
    i=zero(Int64),
    s=UInt8[],
    t=nothing,
    g=nothing,
    sparse_tensor=nothing,
    tp=nothing,
    floats=Vector{Float32}(),
    ints=Vector{Int64}(),
    strings=Vector{Vector{UInt8}}(),
    tensors=Vector{TensorProto}(),
    graphs=Vector{GraphProto}(),
    sparse_tensors=Vector{SparseTensorProto}(),
    type_protos=Vector{TypeProto}(),
)
    return AttributeProto(
        name,
        ref_attr_name,
        doc_string,
        var"#type",
        f,
        i,
        s,
        t,
        g,
        sparse_tensor,
        tp,
        floats,
        ints,
        strings,
        tensors,
        graphs,
        sparse_tensors,
        type_protos,
    )
end
