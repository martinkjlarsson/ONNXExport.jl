struct ExportOptions
    opsets::Vector{OperatorSetIdProto}
end
ExportOptions() = ExportOptions([])

struct GraphContext
    opts::ExportOptions
    nodes::Vector{NodeProto}
    inits::Vector{Union{TensorProto,SparseTensorProto}}
    values::Vector{ValueInfoProto}
end
GraphContext(opts::ExportOptions) = GraphContext(opts, [], [], [])

const GRAPH_CONTEXT = ScopedValue(GraphContext(ExportOptions()))

"""
    has_minimum_opset([opts::ExportOptions,] domain::String, version::Int)

Return whether the provided opset or later is targeted for export.
"""
function has_minimum_opset(opts::ExportOptions, domain::String, version::Int)
    for opset in opts.opsets
        if domain_equal(opset.domain, domain) && opset.version >= version
            return true
        end
    end
    return false
end

function has_minimum_opset(domain::String, version::Int)
    return has_minimum_opset(GRAPH_CONTEXT[].opts, domain, version)
end

function domain_equal(d1::String, d2::String)
    d1 == d2 && return true
    d1 == "" && d2 == "ai.onnx" && return true
    d1 == "ai.onnx" && d2 == "" && return true
    return false
end
