# function MLUtils.batched_searchsortedfirst() end

# function MLUtils.batched_searchsortedlast() end

function MLUtils.chunk(x::ProbeArray, n::Int; dims::Int=ndims(x))
    return with_prefix("chunk") do
        split_dim_size = raw_size(x, dims)
        if split_dim_size isa Int
            size_first = cld(split_dim_size, n)
            size_rest = rem(split_dim_size, size_first)
        else
            size_first = dimension_name()
            size_rest = dimension_name()
        end

        dims_first = ntuple(i -> i == dims ? size_first : raw_size(x, i), ndims(x))
        dims_rest = ntuple(i -> i == dims ? size_rest : raw_size(x, i), ndims(x))

        outputs = ntuple(n) do i
            if i == n
                return value_info(eltype(x), dims_rest, "output")
            else
                return value_info(eltype(x), dims_first, "output")
            end
        end

        onnx_op("Split", x, outputs; attr=(axis=ndims(x)-dims, num_outputs=n))

        return collect(outputs)
    end
end

function MLUtils.chunk(x::ProbeArray; size, dims::Int=ndims(x))
    split_dim_size = raw_size(x, dims)
    if split_dim_size isa Symbol
        error(
            "ONNX export of MLUtils.chunk is not possible when size is provided and dims is a symbolic dimension. Cannot statically deduce the number of outputs.",
        )
    end

    if size isa Int
        # Might as well avoid creating the split tensor.
        n = cld(split_dim_size, size)
        return chunk(x, n; dims=dims)
    end

    @assert sum(size) == split_dim_size

    return with_prefix("chunk") do
        split = probe(collect(Int64, size), "split")

        outputs = ntuple(length(size)) do j
            new_dims = ntuple(i -> i == dims ? size[j] : raw_size(x, i), ndims(x))
            return value_info(eltype(x), new_dims, "output")
        end

        onnx_op("Split", (x, split), outputs; attr=(axis=ndims(x)-dims, num_outputs=n))

        return collect(outputs)
    end
end

function MLUtils.flatten(x::ProbeArray)
    sz = raw_size(x)
    sz1 = sz[1:(end - 1)]
    sz2 = sz[end]

    new_dims = sz1 isa Dims ? (prod(sz1), sz2) : (dimension_name(), sz2)
    return onnx_op("Flatten", new_dims, x)
end

# function MLUtils.group_counts() end

# function MLUtils.group_indices() end

function MLUtils.normalise(x::ProbeArray; dims=ndims(x), ϵ=eltype(x)(1e-5))
    if eltype(x)(ϵ) != eltype(x)(1e-9)
        @warn "Exporting MLUtils.normalise with ϵ=1e-9 to match the ONNX operator MeanVarianceNormalization."
    end

    if dims isa Integer
        attr = (axes=Int[ndims(x) - dims],)
    else
        attr = (axes=collect(Int, ndims(x) .- dims),)
    end

    return onnx_op("MeanVarianceNormalization", x; attr=attr)
end

# There is no dedicated ONNX operation for this. The existing method works.
# function MLUtils.rescale() end

# TODO: Implement when NNlib padding is implemented.
# function MLUtils.rpad_constant() end

function MLUtils.topk(x::ProbeArray, k::Integer; dims::Integer=1, rev::Bool=true)
    new_dims = ntuple(i -> i == dims ? k : raw_size(x, i), ndims(x))

    values = value_info(eltype(x), new_dims, "values")
    indices = value_info(Int64, new_dims, "indices")
    attr=(axis=ndims(x) - dims, largest=rev)

    onnx_op("TopK", (x, probe(Int64[k])), (values, indices); attr=attr)

    # Convert to one-based indices.
    indices = indices .+ 1

    return values, indices
end

# function MLUtils.unbatch() end

function MLUtils.unsqueeze(x::ProbeArray; dims::Int)
    return ONNXExport.unsqueeze(x, dims)
end

# function MLUtils.unstack() end
