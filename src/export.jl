# TODO: Should fullname! be used instead to guarantee uniqueness?
function output_name(node_name)
    return node_name * "_output"
end

"""
Trace the function or callable `f` and get an ONNX (sub)graph.
"""
function graph_function(f, inputs::ProbeArray...; gn="graph")
    ctx = GraphContext()
    gn = graph_name(gn)

    return with_prefix(gn) do
        return with(GRAPH_CONTEXT => ctx) do
            probed_inputs = Tuple(create_input.(inputs))
            outputs = f(probed_inputs...)

            # An output with the same name as an input or other output may cause issues on
            # runtime. Insert Identity operator as needed.
            used_names = Set(name(A) for A in probed_inputs)
            outputs = map(outputs) do output
                if name(output) ∈ used_names
                    return onnx_op("Identity", output)
                end
                push!(used_names, name(output))
                return output
            end

            ivi = value_info_list(probed_inputs)
            ovi = value_info_list(outputs)

            graph = GraphProto(gn, ctx.nodes, ivi, ovi, ctx.inits; value_info=ctx.values)

            return graph, outputs
        end
    end
end

function trace_function(f::Function, inputs...)
    ns = Namespace()
    ctx = GraphContext()

    graph = with(NAMESPACE => ns, GRAPH_CONTEXT => ctx) do
        gn = graph_name("main_graph")
        probed_inputs = Tuple(create_input.(inputs))
        outputs = f(probed_inputs...)
        ivi = value_info_list(probed_inputs)
        ovi = value_info_list(outputs)

        graph = GraphProto(gn, ctx.nodes, ivi, ovi, ctx.inits; value_info=ctx.values)

        for i in eachindex(ovi)
            new = value_name("output")
            rename_value!(graph, ovi[i].name, new)
        end

        return graph
    end

    return graph
end

function create_model(
    f::Function,
    inputs...;
    ir_version=10,
    opset_import=[OperatorSetIdProto("", 21)],
    kwargs...,
)
    graph = trace_function(f, inputs...)
    model = ModelProto(graph; ir_version=ir_version, opset_import=opset_import, kwargs...)

    optimize_dead_ops!(model)

    return model
end

function export_model(file_name::String, f::Function, inputs...; kwargs...)
    onnx_model = create_model(f, inputs...; kwargs...)
    return savemodel(file_name, onnx_model)
end

function export_model(io::IO, f::Function, inputs...; kwargs...)
    onnx_model = create_model(f, inputs...; kwargs...)
    return savemodel(io, onnx_model)
end
