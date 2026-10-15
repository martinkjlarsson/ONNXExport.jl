function output_name(op_type)
    return op_type * "_output"
end

function create_probe(ti::TypeInfo)
    full_name = add_input_value(ti::TypeInfo)
    if isscalar(ti)
        return ProbeNumber{eltype(ti)}(full_name)
    else
        return ProbeArray{eltype(ti)}(full_name, raw_size(ti))
    end
end

function add_input_value(ti::TypeInfo)
    fn = name(ti)
    if isempty(fn) || has_value(fn)
        fn = get_value_name("input")
    else
        add_value(fn)
    end

    ctx = GRAPH_CONTEXT[]
    vi = TensorValueInfoProto(fn, eltype(ti), reverse(raw_size(ti)))
    push!(ctx.values, vi)

    return fn
end

function trace_function(
    f::Function, inputs::TypeInfo...; opts::ExportOptions, output_names=nothing
)
    ns = Namespace()
    return with(NAMESPACE => ns) do
        gn = get_graph_name("main_graph")
        return trace_common(
            f, inputs...; graph_name=gn, opts=opts, output_names=output_names
        )
    end

    return graph
end

function trace_sub_function(
    f::Function, inputs::TypeInfo...; opts::ExportOptions=GRAPH_CONTEXT[].opts
)
    gn = get_graph_name("graph")

    return with_prefix(gn) do
        return trace_common(f, inputs...; graph_name=gn, opts=opts)
    end
end

function trace_common(
    f::Function, inputs::TypeInfo...; graph_name, opts, output_names=nothing
)
    verify_input_output_names(inputs, output_names)

    ctx = GraphContext(opts)

    return with(GRAPH_CONTEXT => ctx) do
        inputs = create_probe.(inputs)
        outputs = f(inputs...)
        outputs = as_tuple(outputs)
        outputs = probe(outputs)

        for out in outputs
            if !isprobe(out)
                error(
                    "The traced function returned an argument of type $(typeof(out)), " *
                    "which cannot be converted to an ONNX tensor.",
                )
            end
        end

        # An output with the same name as an input or other output may cause issues on
        # runtime. Insert Identity operator as needed.
        used_names = Set{String}(name(A) for A in inputs)
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
        if isnothing(output_names)
            for i in eachindex(ovi)
                new_name = get_value_name("output")
                rename_value!(graph, ovi[i].name, new_name)
            end
        elseif length(output_names) == length(ovi)
            for (i, new_name) in enumerate(output_names)
                if has_value(new_name)
                    throw(
                        ArgumentError(
                            "The output name \"$new_name\" already exists in the graph."
                        ),
                    )
                end
                rename_value!(graph, ovi[i].name, new_name)
            end
        else
            throw(
                ArgumentError(
                    "The length of output_names must match the number of outputs. Got " *
                    "$(length(output_names)) names but the model has $(length(ovi)) " *
                    "outputs.",
                ),
            )
        end

        return graph, outputs
    end
end

function verify_input_output_names(inputs, output_names) where {N}
    input_names = Set{String}()
    for input in inputs
        fn = name(input)
        if isempty(fn)
            continue
        end
        if fn ∈ input_names
            error("Input argument names cannot repeat. \"$fn\" is used multiple times.")
        end
        push!(input_names, fn)
    end

    if !isnothing(output_names)
        if length(unique(output_names)) != length(output_names)
            throw(ArgumentError("output_names cannot contain duplicate entries."))
        end
        if any(isempty.(output_names))
            throw(ArgumentError("output_names cannot contain empty entires."))
        end
    end
end

"""
    trace(f::Function, inputs...; kwargs...)

Trace the function `f` called with the arguments `inputs...` and create an ONNX model.

The input arguments can be any `AbstractArray{<:Number}` or `Number`, provided the element
type has a corresponding ONNX tensor data type. This is true for most Julia `Number`s. The
arguments are only used to infer the element type and size. The values themselves are not
used. Alternative, the input arguments can be instances of `TypeInfo`, which enables
specifying symbolic dimensions (dimensions of unknown size) and providing input names.

# Arguments
- `output_names::Union{Nothing,NTuple{NO,String}}=nothing`: names to assign to the output
  nodes of the graph. If `nothing`, the names default to "output1", "output2", etc.
- `opset_version::Int=21`: the version of the default ai.onnx opset to target. Later
  versions allow for newer ONNX operators and possibly better performance but may not be
  supported by certain runtimes. This argument does nothing if `opset_import` is provided.

Additional arguments passed to the created `ModelProto` (see the
[ONNX docs](https://onnx.ai/onnx/repo-docs/IR.html#models) and the
[proto file](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3#L446) for details):
- `ir_version::Integer=10`
- `opset_import::Vector{OperatorSetIdProto}=[OperatorSetIdProto("", opset_version)]`
- `producer_name::String=<set by ONNXHelper>`
- `producer_version::String<set by ONNXHelper>`
- `domain::String=""`
- `model_version::Integer=0`
- `doc_string::String=""`
- `metadata_props::Vector{StringStringEntryProto}=[]`
- `training_info::Vector{TrainingInfoProto}=[]`
- `configuration::Vector{DeviceConfigurationProto}=[]`

# Examples
```julia
using ONNXExport

f(x, y) = x .+ y .- 3
model = ONNXExport.trace(f, rand(Float32, 3, 4), rand(Float32, 3))
ONNXExport.save("model.onnx", model)
```

See also [`save`](@ref).
"""
function trace(
    f::Function,
    inputs::Union{AbstractArray{<:Number},Number,TypeInfo}...;
    output_names::Union{Nothing,NTuple{NO,String}}=nothing,
    opset_version::Int=21,
    ir_version=10,
    opset_import=[OperatorSetIdProto("", opset_version)],
    producer_name=string(ONNXExport) * ".jl",
    producer_version=string(pkgversion(ONNXExport)),
    kwargs...,
) where {NO}
    opts = ExportOptions(opset_import)

    if !has_minimum_opset(opts, "", 21)
        @warn "Targeting an opset version lower than 21 is untested and may result in an " *
            "invalid model."
    end
    lastest_ir_version = Int(ONNXHelper.Version.IR_VERSION)
    if ir_version > lastest_ir_version
        throw(
            ArgumentError(
                "The requested IR version ($ir_version) is newer than the lastest " *
                "supported version ($lastest_ir_version).",
            ),
        )
    end

    inputs = type_info.(inputs)
    graph, _ = trace_function(f, inputs...; opts, output_names)
    model = ModelProto(
        graph; ir_version, opset_import, producer_name, producer_version, kwargs...
    )

    ONNXHelper.optimize_dead_ops!(model)

    return model
end

"""
    save(file_name::String, f::Function, inputs...; kwargs...)
    save(io::IO, f::Function, inputs...; kwargs...)

Trace and save a Julia function as an ONNX model.

Create an ONNX model from the function `f` called with the arguments `inputs...`, and save
it to the provide `IO` or `String` file name. See [`trace`](@ref) for details.

See also [`trace`](@ref).
"""
function save(file_name::String, f::Function, inputs...; kwargs...)
    onnx_model = trace(f, inputs...; kwargs...)
    return save(file_name, onnx_model)
end

function save(io::IO, f::Function, inputs...; kwargs...)
    onnx_model = trace(f, inputs...; kwargs...)
    return save(io, onnx_model)
end

"""
    save(file_name::String, model::ModelProto)
    save(io::IO, model::ModelProto)

Save a traced ONNX model to file or IO.

See also [`trace`](@ref).
"""
function save(file_name::String, model::ModelProto)
    return ONNXHelper.save(file_name, model)
end

function save(io::IO, model::ModelProto)
    return ONNXHelper.save(io, model)
end
