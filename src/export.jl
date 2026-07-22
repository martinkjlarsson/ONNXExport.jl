# TODO: Should fullname! be used instead to guarantee uniqueness?
function output_name(node_name)
    return node_name * "_output"
end

astuple(x::Tuple) = x
astuple(x) = (x,)

template_to_probe(A::ProbeArray) = A
template_to_probe(A::AbstractArray) = ProbeArray{eltype(A)}("", size(A))
template_to_probe(x::ProbeNumber) = x
template_to_probe(x::Number) = ProbeNumber{typeof(x)}("")
function template_to_probe(::T) where {T}
    return error("Input argument of type $T cannot be converted to an ONNX tensor.")
end

function add_input_value(A::Union{ProbeArray{T},ProbeNumber{T}}) where {T}
    fn = name(A)
    if isempty(fn) || has_value(fn)
        fn = get_value_name("input")
    else
        add_value(fn)
    end

    ctx = GRAPH_CONTEXT[]
    vi = TensorValueInfoProto(fn, T, reverse(raw_size(A)))
    push!(ctx.values, vi)

    return fn
end

function create_input(A::ProbeArray{T}) where {T}
    fn = add_input_value(A)
    return ProbeArray{T}(fn, raw_size(A))
end
function create_input(A::ProbeNumber{T}) where {T}
    fn = add_input_value(A)
    return ProbeNumber{T}(fn)
end

function trace_function(f::Function, inputs::Union{AbstractArray{<:Number},Number}...)
    return trace_function(f, template_to_probe.(inputs)...)
end

function trace_function(f::Function, inputs::Union{ProbeArray,ProbeNumber}...)
    names = Set{String}()
    for input in inputs
        fn = name(input)
        if isempty(fn)
            continue
        end
        if fn ∈ names
            error("Input argument names cannot repeat. \"$fn\" is used multiple times.")
        end
        push!(names, fn)
    end

    ns = Namespace()
    return with(NAMESPACE => ns) do
        gn = get_graph_name("main_graph")
        return trace_common(f, inputs...; graph_name=gn)
    end

    return graph
end

function trace_sub_function(f::Function, inputs::Union{ProbeArray,ProbeNumber}...)
    gn = get_graph_name("graph")

    return with_prefix(gn) do
        return trace_common(f, inputs...; graph_name=gn)
    end
end

function trace_common(f::Function, inputs::Union{ProbeArray,ProbeNumber}...; graph_name)
    ctx = GraphContext()

    return with(GRAPH_CONTEXT => ctx) do
        inputs = create_input.(inputs)
        outputs = f(inputs...)
        outputs = astuple(outputs)
        outputs = probe(outputs)

        for out in outputs
            if !isprobe(out)
                error(
                    "The traced function returned an argument of type $(typeof(out)), which cannot be converted to an ONNX tensor.",
                )
            end
        end

        # An output with the same name as an input or other output may cause issues on
        # runtime. Insert Identity operator as needed.
        used_names = Set(name(A) for A in inputs)
        outputs = map(outputs) do output
            if name(output) ∈ used_names
                return onnx_op("Identity", output)
            end
            push!(used_names, name(output))
            return output
        end

        ivi = value_info_list(inputs)
        ovi = value_info_list(outputs)

        graph = GraphProto(
            graph_name, ctx.nodes, ivi, ovi, ctx.inits; value_info=ctx.values
        )

        # Rename outputs.
        for i in eachindex(ovi)
            new = get_value_name("output")
            rename_value!(graph, ovi[i].name, new)
        end

        return graph, outputs
    end
end

"""
    create_model(f::Function, inputs...; kwargs...)

Create an ONNX model from the function `f` called with the arguments `inputs...`.

The input arguments can be any `AbstractArray{<:Number}` or `Number`, provided the element
type has a corresponding ONNX tensor data type. This is true for most Julia `Number`s. The
arguments are only used to infer the element type and size. The values themselves are not
used. Use `ProbeArray` or `ProbeNumber` to name the inputs or to provide symbolic
dimensions.

# Arguments
See the [ONNX docs](https://onnx.ai/onnx/repo-docs/IR.html#models) and the
[proto file](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3#L446) for a
description of the arguments.
- `ir_version::Integer=10`
- `opset_import::Vector{OperatorSetIdProto}=[OperatorSetIdProto("", 21)]`
- `producer_name::String=<set by ONNXHelper>`
- `producer_version::String<set by ONNXHelper>`
- `domain::String=""`
- `model_version::Integer=0`
- `doc_string::String=""`
- `metadata_props::Vector{StringStringEntryProto}=[]`
- `training_info::Vector{TrainingInfoProto}=[]`
- `configuration::Vector{DeviceConfigurationProto}=[]`
"""
function create_model(
    f::Function,
    inputs::Union{AbstractArray{<:Number},Number}...;
    ir_version=10,
    opset_import=[OperatorSetIdProto("", 21)],
    kwargs...,
)
    graph, _ = trace_function(f, inputs...)
    model = ModelProto(graph; ir_version=ir_version, opset_import=opset_import, kwargs...)

    ONNXHelper.optimize_dead_ops!(model)

    return model
end

"""
    export_model(file, f::Function, inputs...; kwargs...)

Trace and export a Julia function as an ONNX model.

Create an ONNX model from the function `f` called with the arguments `inputs...`, and save
it to the provide `IO` or `String` file name. See [`create_model`](@ref) for details.

See also [`create_model`](@ref), [`trace_function`](@ref).
"""
function export_model(file_name::String, f::Function, inputs...; kwargs...)
    onnx_model = create_model(f, inputs...; kwargs...)
    return save_model(file_name, onnx_model)
end

function export_model(io::IO, f::Function, inputs...; kwargs...)
    onnx_model = create_model(f, inputs...; kwargs...)
    return save_model(io, onnx_model)
end
