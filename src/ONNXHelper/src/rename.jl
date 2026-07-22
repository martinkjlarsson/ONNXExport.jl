function rename_value!(graph::GraphProto, old::String, new::String)
    # Rename nodes.
    for node in graph.node
        rename_value!(node, old, new)
    end

    # Rename inputs and outputs.
    for i in eachindex(graph.input)
        graph.input[i] = rename_value(graph.input[i], old, new)
    end
    for i in eachindex(graph.output)
        graph.output[i] = rename_value(graph.output[i], old, new)
    end

    # Rename initializers.
    for i in eachindex(graph.initializer)
        graph.initializer[i] = rename_value(graph.initializer[i], old, new)
    end
    for i in eachindex(graph.sparse_initializer)
        graph.sparse_initializer[i] = rename_value(graph.sparse_initializer[i], old, new)
    end

    # Rename values.
    for i in eachindex(graph.value_info)
        graph.value_info[i] = rename_value(graph.value_info[i], old, new)
    end

    return nothing
end

function rename_value!(node::NodeProto, old::String, new::String)
    # Rename inputs and outputs.
    replace!(node.input, old => new)
    replace!(node.output, old => new)

    # Rename attributes.
    for attr in node.attribute
        if attr.var"#type" == var"AttributeProto.AttributeType".TENSOR

        elseif attr.var"#type" == var"AttributeProto.AttributeType".TENSORS
        elseif attr.var"#type" == var"AttributeProto.AttributeType".GRAPH
            rename_value!(attr.g, old, new)
        elseif attr.var"#type" == var"AttributeProto.AttributeType".GRAPHS
            for g in attr.graphs
                rename_value!(g, old, new)
            end
        end
    end
end

function rename_value(attr::AttributeProto, old::String, new::String)
    if attr.var"#type" == var"AttributeProto.AttributeType".TENSOR
        t = rename_value(attr.t, old, new)
        if t !== attr.t
            return AttributeProto(
                attr.name,
                attr.ref_attr_name,
                attr.doc_string,
                attr.var"#type",
                attr.f,
                attr.i,
                attr.s,
                t,
                attr.g,
                attr.sparse_tensor,
                attr.tp,
                attr.floats,
                attr.ints,
                attr.strings,
                attr.tensors,
                attr.graphs,
                attr.sparse_tensors,
                attr.type_protos,
            )
        end
    elseif attr.var"#type" == var"AttributeProto.AttributeType".TENSORS
        for i in eachindex(attr.tensors)
            attr.tensors[i] = rename_value(attr.tensors[i], old, new)
        end
    elseif attr.var"#type" == var"AttributeProto.AttributeType".GRAPH
        rename_value!(attr.g, old, new)
    elseif attr.var"#type" == var"AttributeProto.AttributeType".GRAPHS
        for g in attr.graphs
            rename_value!(g, old, new)
        end
    end
    return attr
end

function rename_value(tensor::TensorProto, old::String, new::String)
    if tensor.name != old
        return tensor
    end

    return TensorProto(
        tensor.dims,
        tensor.data_type,
        tensor.segment,
        tensor.float_data,
        tensor.int32_data,
        tensor.string_data,
        tensor.int64_data,
        new,
        tensor.doc_string,
        tensor.raw_data,
        tensor.external_data,
        tensor.data_location,
        tensor.double_data,
        tensor.uint64_data,
        tensor.metadata_props,
    )
end

function rename_value(tensor::SparseTensorProto, old::String, new::String)
    # The values field is used to name the sparse tensor.
    t = rename_value(tensor.values, old, new)
    if t === tensors.values
        return tensor
    end

    return SparseTensorProto(t, tensor.indices, tensor.dims)
end

function rename_value(value::ValueInfoProto, old::String, new::String)
    if value.name != old
        return value
    end

    return ValueInfoProto(new, value.var"#type", value.doc_string, value.metadata_props)
end
