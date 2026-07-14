const AnyProbe{T} = Union{ProbeArray{T},ProbeNumber{T},BroadcastProbe{T}}

"""
    onnx_op(
        op_type,
        [outtype=eltype(first(inputs))],
        [dims=broadcast_shape(inputs...)],
        inputs...;
        attr=[],
        ctx=get_context(inputs...),
        domain="",
    )
    onnx_op(op_type, inputs, outputs; attr=[], ctx=get_context(inputs...), domain="")

Create an ONNX operator of the specified `op_type`, taking certain `inputs` and returning
the `outputs`. If there is only one output, its type and size can optionally be set using
`outtype` and `dims`. For multiple outputs, tuples of `inputs` and `outputs` must be
provided, where the outputs are created with `create_value_info!(...)` prior to calling
this function.

Attributes can be provided as any collection implementing `pairs`, e.g., `NamedTuple` and
`Dict`, or as a `AbstractVector{AttributeProto}`.
"""
function onnx_op(
    op_type::String,
    outtype::Type,
    outdims::ProbeDims,
    inputs::AnyProbe...;
    attr=AttributeProto[],
    domain="",
)
    ctx = GRAPH_CONTEXT[]
    nn = get_node_name(op_type)
    output = value_info(outtype, outdims, output_name(op_type))
    n = NodeProto(op_type, [name.(inputs)...], [name(output)], attr; name=nn, domain=domain)
    push!(ctx.nodes, n)

    # TODO: This is a bit hacky. Fix.
    if outdims == ()
        return ProbeNumber(output)
    elseif any(x -> isa(x, BroadcastProbe), inputs)
        return BroadcastProbe(output)
    else
        return output
    end
end
function onnx_op(op_type::String, dims::ProbeDims, inputs::AnyProbe...; kwargs...)
    return onnx_op(op_type, eltype(inputs[1]), dims, inputs...; kwargs...)
end
function onnx_op(op_type::String, outtype::Type, inputs::AnyProbe...; kwargs...)
    return onnx_op(op_type, outtype, broadcast_shape(inputs...), inputs...; kwargs...)
end
function onnx_op(op_type::String, inputs::AnyProbe...; kwargs...)
    return onnx_op(
        op_type, eltype(inputs[1]), broadcast_shape(inputs...), inputs...; kwargs...
    )
end

function onnx_op(
    op_type::String, input::AnyProbe, outputs::NTuple{Nout,AnyProbe}; kwargs...
) where {Nout}
    return onnx_op(op_type, (input,), outputs; kwargs...)
end
function onnx_op(
    op_type::String,
    inputs::NTuple{Nin,AnyProbe},
    outputs::NTuple{Nout,AnyProbe};
    attr=AttributeProto[],
    domain="",
) where {Nin,Nout}
    ctx = GRAPH_CONTEXT[]

    nn = get_node_name(op_type)
    n = NodeProto(
        op_type, [name.(inputs)...], [name.(outputs)...], attr; name=nn, domain=domain
    )
    push!(ctx.nodes, n)
    return outputs
end
