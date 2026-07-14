function Base.ifelse(
    condition::AbstractProbeNumber{Bool}, x::T, y::AbstractProbeNumber{T}
) where {T}
    return ifelse(condition, probe(x), y)
end
function Base.ifelse(
    condition::AbstractProbeNumber{Bool}, x::AbstractProbeNumber{T}, y::T
) where {T}
    return ifelse(condition, x, probe(y))
end
function Base.ifelse(
    condition::AbstractProbeNumber{Bool},
    x::AbstractProbeNumber{T},
    y::AbstractProbeNumber{T},
) where {T}
    return onnx_op("Where", T, condition, x, y)
end

"""
    scan_onnx(f, initial_state::NTuple{N,ProbeArray}, scan_inputs::NTuple{M,ProbeArray}; kwargs...)

Export the ONNX [Scan](https://onnx.ai/onnx/operators/onnx__Scan.html) operator.

`f` must return `N+K` `ProbeArray`s, where `N` is the length of the state and `K` is the
length of the scan output. The outputs must have the same type and size in each iteration.
`f` may capture variables from an outer scope but must not have any side effects.

# Keywords
- `scan_input_axes::NTuple{M,<:Integer}=ndims.(scan_inputs)`
- `scan_input_directions::NTuple{M,Bool}=ntuple(Returns(false), M)`
- `scan_output_axes::Union{Nothing,NTuple{K,<:Integer}}=nothing`
- `scan_output_directions::Union{Nothing,NTuple{K,Bool}}=nothing`
"""
function scan_onnx(
    f,
    initial_state::NTuple{N,ProbeArray},
    scan_inputs::NTuple{M,ProbeArray};
    scan_input_axes::NTuple{M,<:Integer}=ndims.(scan_inputs),
    scan_input_directions::NTuple{M,Bool}=ntuple(Returns(false), M),
    scan_output_axes=nothing,
    scan_output_directions=nothing,
) where {M,N}
    # TODO: Replace all asserts with thrown errors instead?
    @assert M > 0 "Scan needs at least one scan input"

    seq_lens = size.(scan_inputs, scan_input_axes)
    @assert allequal(seq_lens) "scan inputs have mismatched sizes along dimensions $scan_input_axes; got sizes $seq_lens"
    seq_len = seq_lens[1]

    # Trace f as a subgraph.
    scan_input_elts = ntuple(M) do i
        A = scan_inputs[i]
        return ProbeArray{eltype(A)}("", _selectdimsize(raw_size(A), scan_input_axes[i]))
    end
    graph, outputs = trace_sub_function(f, initial_state..., scan_input_elts...)

    @assert length(outputs) >= N "f must have at least N=$N outputs; got $(length(outputs))"

    graph_state_outputs = outputs[1:N]
    graph_scan_output_elts = outputs[(N + 1):end]
    K = length(graph_scan_output_elts)

    # Verify outputs.
    for (i, init, output) in zip(1:N, initial_state, graph_state_outputs)
        @assert eltype(init) == eltype(output) "element type of state output $i must match initial state; expected $(eltype(init)) but got $(eltype(output))"
        if raw_size(init) != raw_size(output)
            # TODO: Throw error if static dimensions are mismatched.
            @warn "size of state output $i does not match initial state; expected $(raw_size(init)) but got $(raw_size(output))"
        end
    end

    if isnothing(scan_output_axes)
        scan_output_axes = ndims.(scan_output_axes)
    end
    if isnothing(scan_output_directions)
        scan_output_directions = ntuple(Returns(false), K)
    end

    @assert length(scan_output_axes) == K "scan_output_axes has incorrect length; expected $K but got $(length(scan_output_axes))"
    @assert length(scan_output_directions) == K "scan_output_directions has incorrect length; expected $K but got $(length(scan_input_directions))"

    # Create Scan outputs.
    state_outputs = [
        value_info(eltype(A), raw_size(A), "state_output") for A in graph_state_outputs
    ]
    scan_outputs = ntuple(K) do i
        A = graph_scan_output_elts[i]
        return value_info(
            eltype(A),
            _stackdimsize(raw_size(A), scan_output_axes[i], seq_len),
            "scan_output",
        )
    end

    onnx_op(
        "Scan",
        (initial_state..., scan_inputs...),
        (state_outputs..., scan_outputs...);
        attr=(
            body=graph,
            num_scan_inputs=M,
            scan_input_axes=ndims.(scan_inputs) .- scan_input_axes,
            scan_input_directions=scan_input_directions,
            scan_output_axes=ndims.(scan_outputs) .- scan_output_axes,
            scan_output_directions=scan_output_directions,
        ),
    )

    return state_outputs, scan_outputs
end

_selectdimsize(sz::ProbeDims, dim) = (sz[1:(dim - 1)]..., sz[(dim + 1):end]...)
function _stackdimsize(sz::ProbeDims, dim, count)
    if dim <= length(sz) + 1
        return (sz[1:(dim - 1)]..., count, sz[dim:end]...)
    end
    throw(
        ArgumentError(
            "cannot stack scan ouput slices A, where ndims(A) = $(length(sz)) along dimension $dim",
        ),
    )
end

# function loop_onnx(f, n::Int, cond::Bool, args...)
#     i = 1
#     args_in = args
#     while cond
#         if i > n
#             break
#         end
#         cond, args_in, scan_out = f(i, cond, args_in...)
#         i += 1
#     end
#     # TODO: Stack the outputs from scan_out
#     return args_in, scan_out_stacked
# end

# function if_onnx(then_branch, condition::Bool, else_results, args...)
#     return condition ? then_branch(args...) : else_results
# end
# function if_onnx(condition::Bool, then_branch, else_branch, args...)
#     return condition ? then_branch(args...) : else_branch(args...)
# end

# function if_onnx(condition::ProbeScalar{Bool}, then_branch, else_branch, args...)
#     ctx = condition.ctx

#     then_result = then_branch(args...)
#     else_result = else_branch(args...)

#     # TODO: Call promote_probe on results using ctx above.

#     if typeof(then_result) == typeof(else_result)
#         error(
#             "Return types of then branch and else branch are mismatched; got $(typeof(then_result)) and $(typeof(else_result))",
#         )
#     end

#     outputs = if_outputs(then_result, else_result)
#     return onnx_op("If", condition; outputs=outputs) # TODO: Set branch graphs.
# end

# function if_outputs(then_result::Tuple, else_result::Tuple)
#     return if_outputs.(then_result, else_result)
# end
# function if_outputs(then_result::ProbeArray, else_result::ProbeArray)
#     @assert size(then_result) == size(else_result) "Dynamic dimensions is not yet supported"
#     fname = fullname!(then_result.ctx, output_name("If"))
#     return value_info(then_result, fname)
# end
