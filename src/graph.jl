struct GraphContext
    nodes::Vector{NodeProto}
    inits::Vector{Union{TensorProto,SparseTensorProto}}
    values::Vector{ValueInfoProto}
end
GraphContext() = GraphContext([], [], [])

const GRAPH_CONTEXT = ScopedValue(GraphContext())
