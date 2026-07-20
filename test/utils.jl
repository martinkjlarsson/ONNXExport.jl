# ONNXRunTime permutes all dimensions so we must undo this.
function prepare_input(A::AbstractArray)
    return PermutedDimsArray(A, ntuple(i -> ndims(A) + 1 - i, ndims(A)))
end
function prepare_input(scalar)
    a = Array{typeof(scalar),0}(undef)
    a[] = scalar
    return a
end
function prepare_output(A::AbstractArray)
    return permutedims(A, ntuple(i -> ndims(A) + 1 - i, ndims(A)))
end
function prepare_output(scalar::AbstractArray{T,0}) where {T}
    return scalar[]
end

function test_function(f::Function, x...)
    # Evaluate function in Julia.
    y = f(x...)

    # Save function to ONNX.
    file_name = tempname() * ".onnx"
    onnx_model = create_model(f, x...)
    save_model(file_name, onnx_model)
    @info "Saved model for $f at $file_name"

    # Evaluate ONNX function.
    input_names = [vi.name for vi in onnx_model.graph.input]
    input_dict = Dict(input_names[i] => prepare_input(x[i]) for i in eachindex(x))

    path = ONNXRunTime.testdatapath(file_name)
    model = load_inference(path)
    output_dict = model(input_dict)

    if y isa Tuple
        y_onnx = tuple(prepare_output(v) for v in values(output_dict))
    else
        y_onnx = prepare_output(first(values(output_dict)))
    end

    return y, y_onnx
end

function test_model(rng, model, x; test=false)
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
    onnx_model = create_model(f, input...)
    save_model(file_name, onnx_model)
    @info "Saved model at $file_name"

    # Evaluate ONNX model.
    input_names = [vi.name for vi in onnx_model.graph.input]
    input_dict = Dict(zip(input_names, prepare_input.(input)))

    model = load_inference(file_name)
    output_dict = model(input_dict)

    if y isa Tuple
        y_onnx = tuple(prepare_output(v) for v in values(output_dict))
    else
        y_onnx = prepare_output(first(values(output_dict)))
    end

    return y, y_onnx
end
