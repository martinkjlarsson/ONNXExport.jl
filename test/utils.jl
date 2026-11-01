const OPSET_VERSION = 21

prepare_input(A::AbstractArray) = A
prepare_input(scalar::Number) = fill(scalar)
prepare_output(A::AbstractArray) = A
prepare_output(scalar::AbstractArray{T,0}) where {T} = scalar[]

function test_function(f::Function, x...; evaluate=true, opset_version=OPSET_VERSION)
    # Evaluate function in Julia.
    y = f(x...)

    # Save function to ONNX.
    file_name = tempname() * ".onnx"
    opsets = [
        ONNXExport.OperatorSetIdProto("", opset_version),
        ONNXExport.OperatorSetIdProto("com.microsoft", 1),
    ]
    onnx_model = ONNXExport.trace(f, x...; opset_import=opsets)
    ONNXExport.save(file_name, onnx_model)
    @info "Saved model for $f at $file_name"

    if !evaluate
        return nothing
    end

    # Evaluate ONNX function.
    input_names = [vi.name for vi in onnx_model.graph.input]
    input_dict = Dict(input_names[i] => prepare_input(x[i]) for i in eachindex(x))

    model = load_inference(file_name; reverse_dims=false)
    output_dict = model(input_dict)

    if y isa Tuple
        y_onnx = tuple([prepare_output(v) for v in values(output_dict)]...)
    else
        y_onnx = prepare_output(first(values(output_dict)))
    end

    return y, y_onnx
end

"Replace the last dimension of multidimensional arrays with a symbolic dimension."
function make_batched(A)
    ndims(A) <= 1 && return A
    return TypeInfo(eltype(A), size(A)[1:(end - 1)]..., :N)
end

function test_model(
    rng,
    model,
    x;
    test=false,
    evaluate=true,
    opset_version=OPSET_VERSION,
    symbolic_batch=true,
)
    ps, st = Lux.setup(rng, model)
    if test
        st = Lux.testmode(st)
    end

    # Evaluate model in Julia.
    y, _ = Lux.apply(model, x, ps, st)

    # Save model to ONNX.
    file_name = tempname() * ".onnx"
    if x isa Tuple
        input = x
        f = (x...) -> first(Lux.apply(model, x, ps, st))
    else
        input = (x,)
        f = x -> first(Lux.apply(model, x, ps, st))
    end
    symbolic_input = symbolic_batch ? make_batched.(input) : input
    opsets = [
        ONNXExport.OperatorSetIdProto("", opset_version),
        ONNXExport.OperatorSetIdProto("com.microsoft", 1),
    ]
    onnx_model = ONNXExport.trace(f, symbolic_input...; opset_import=opsets)
    ONNXExport.save(file_name, onnx_model)
    @info "Saved model at $file_name"

    if !evaluate
        return nothing
    end

    # Evaluate ONNX model.
    input_names = [vi.name for vi in onnx_model.graph.input]
    input_dict = Dict(zip(input_names, prepare_input.(input)))

    model = load_inference(file_name; reverse_dims=false)
    output_dict = model(input_dict)

    if y isa Tuple
        y_onnx = tuple([prepare_output(v) for v in values(output_dict)]...)
    else
        y_onnx = prepare_output(first(values(output_dict)))
    end

    return y, y_onnx
end
