as_tuple(x::Tuple) = x
as_tuple(x) = (x,)

expand_tuple(i::Tuple, _) = i
expand_tuple(i, n) = ntuple(Returns(i), n)

pad_tuple(::Nothing, n) = ntuple(Returns(1), n)
pad_tuple(i::Number, n) = (i, ntuple(Returns(1), n - 1)...)
pad_tuple(itr, n) = (itr..., ntuple(Returns(1), n - length(itr))...)

function interleave_tuples(a::NTuple{N}, b::NTuple{N}) where {N}
    return ntuple(i -> isodd(i) ? a[(i + 1) ÷ 2] : b[i ÷ 2], 2*N)
end
