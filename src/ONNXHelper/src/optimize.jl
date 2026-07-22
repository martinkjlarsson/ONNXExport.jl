function optimize_dead_ops!(model::ModelProto)
    n_removed = graph_dead_ops!(Set{String}(), model.graph)

    for func in model.functions
        n_removed += func_dead_ops!(Set{String}(), func)
    end

    @debug "Optimizing ONNX model: Removed $n_removed dead nodes"

    return nothing
end

function graph_dead_ops!(names, graph::GraphProto)
    return dead_ops!(
        names,
        graph.node,
        graph.output,
        graph.initializer,
        graph.sparse_initializer,
        graph.value_info,
    )
end
function func_dead_ops!(names, func::FunctionProto)
    return dead_ops!(
        names, func.node, func.output, TensorProto[], SparseTensorProto[], func.value_info
    )
end
function dead_ops!(names, nodes, outputs, inits, sparse_inits, values)
    n_removed = 0

    while true
        # Add all values, i.e., initializers and node outputs, to the set of names. Remove
        # any values used as node inputs from the set. The names left in the set are
        # "dead" and should be removed from the graph/function.
        for init in inits
            push!(names, init.name)
        end
        for init in sparse_inits
            push!(names, init.values.name)
        end

        for node in nodes
            setdiff!(names, node.input)
            union!(names, node.output)

            for attr in node.attribute
                if attr.var"#type" == var"AttributeProto.AttributeType".GRAPH
                    n_removed += graph_dead_ops!(names, attr.g)
                elseif attr.var"#type" == var"AttributeProto.AttributeType".GRAPHS
                    for g in attr.graphs
                        n_removed += graph_dead_ops!(names, g)
                    end
                end
            end
        end
        for output in outputs
            delete!(names, output.name)
        end

        if isempty(names)
            break
        end

        # Remove nodes.
        to_remove = Int[]
        for (i, node) in enumerate(nodes)
            # Remove node if none of its outputs have been used elsewhere.
            if all(∈(names), node.output)
                push!(to_remove, i)
            end
        end
        deleteat!(nodes, to_remove)
        filter!(init -> init.name ∉ names, inits)
        filter!(init -> init.values.name ∉ names, sparse_inits)
        filter!(value -> value.name ∉ names, values)

        if isempty(to_remove)
            break
        end
        n_removed += length(to_remove)
    end

    return n_removed
end
