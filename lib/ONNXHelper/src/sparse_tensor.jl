function SparseTensorProto(
    array::AbstractArray; name="", doc_string="", metadata_props=StringStringEntryProto[]
)
    lin_idxs, v = findnonzero(array)
    lin_idxs .-= 1 # Zero-based indices.

    @assert issorted(lin_idxs) "Linear indices must be sorted in ascending order"

    indices = TensorProto(lin_idxs)
    values = TensorProto(v; name=name, doc_string=doc_string, metadata_props=metadata_props)

    return SparseTensorProto(values, indices, collect(reverse(size(array))))
end

function findnonzero(array::AbstractSparseArray)
    I..., V = findnz(array)
    CI = CartesianIndex.(I...)
    lin_idxs = LinearIndices(array)[CI]
    return lin_idxs, V
end
function findnonzero(array::AbstractArray)
    I = findall(!iszero, array)
    V = array[I]
    lin_idxs = LinearIndices(array)[I]
    return lin_idxs, V
end

function to_array(st::SparseTensorProto)
    values = to_array(st.values)
    indices = to_array(st.indices) .+ 1 # Convert to one-based indexing.

    array = zeros(eltype(values), reverse(st.dims)...)
    array[indices] = values

    return array
end

function to_sparse_array(st::SparseTensorProto)
    values = to_array(st.values)
    indices = to_array(st.indices) .+ 1 # Convert to one-based indexing.

    array = spzeros(eltype(values), reverse(st.dims)...)
    array[indices] = values

    return array
end
