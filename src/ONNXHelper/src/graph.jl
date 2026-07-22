function GraphProto(
    name,
    node::Vector{NodeProto},
    input,
    output,
    init=TensorProto[];
    doc_string="",
    value_info=ValueInfoProto[],
    quantization_annotation=TensorAnnotation[],
    metadata_props=StringStringEntryProto[],
)
    initializer = filter(t -> t isa TensorProto, init)
    sparse_initializer = filter(t -> t isa SparseTensorProto, init)

    return GraphProto(
        node,
        name,
        initializer,
        sparse_initializer,
        doc_string,
        input,
        output,
        value_info,
        quantization_annotation,
        metadata_props,
    )
end
