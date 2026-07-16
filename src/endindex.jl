"""
Represents the `end` keyword when used in indexing.

Several ONNX operators, e.g., Slice, Gather, and Scatter, indexes into tensors. They allow
negative indices, where `-1` maps to the last index, `-2` to the second last, and so on.

In Julia, a vector `v` of length 10 being indexed with `v[8]` and `v[end-2]` will in both
cases result in the call `Base.getindex(v, 8)`. That is, the precense of `end` is lost.
`EndIndex` solves this and makes the negative index accessible through the `neg_index()`
function. Since Julia uses 1-based indexing, `0` maps to the last element, `-1` to the
second last, and so on.

Note that `EndIndex` only supports addition and subtraction with `Integer`. If any other
operations or types are used, `EndIndex` will revert to an `Int` and the "endness" is
lost.

# Example
```
struct MyArray <: AbstractVector{Nothing} end

Base.size(A::MyArray) = (10,)
Base.lastindex(A::MyArray) = EndIndex(length(A), true)

function Base.getindex(A::MyArray, i)
    @show i, neg_index(i), typeof(i)
    return nothing
end

A = MyArray()
A[8]            # Output: (i, neg_index(i), typeof(i)) = (8, 8, Int64)
A[end - 2]      # Output: (i, neg_index(i), typeof(i)) = (8, -2, EndIndex)
A[2*end - 12]   # Output: (i, neg_index(i), typeof(i)) = (8, 8, Int64)
```
"""
struct EndIndex <: Integer
    i::Int
    offset::Int
end

function EndIndex(::EndIndex)
    return error(
        "Constructing an EndIndex with an EndIndex is ambiguous and thus not permitted."
    )
end
EndIndex(i::Number, isend=false) = isend ? EndIndex(i, i) : EndIndex(i, 0)

"""
    neg_index(x)

To be called inside Base.getindex(...). If `x` is a normal index, this function returns
`x`. If `x` is an `EndIndex` resulting from `end - i`, this functions returns `-i`.
"""
neg_index(x::EndIndex) = x.i - x.offset
neg_index(x::Number) = x

Base.:+(a::EndIndex, b::Integer) = EndIndex(a.i + b, a.offset)
Base.:+(a::Integer, b::EndIndex) = EndIndex(a + b.i, b.offset)
Base.:-(a::EndIndex, b::Integer) = EndIndex(a.i - b, a.offset)
Base.:-(a::EndIndex) = -a.i
Base.:~(a::EndIndex) = ~a.i

for f in
    (:+, :-, :*, :/, :÷, :\, :^, :%, :&, :|, :⊻, :⊼, :⊽, :>>>, :>>, :<<) ∪
    (:(==), :!=, :<, :<=, :>, :>=)

    @eval Base.$f(a::EndIndex, b::EndIndex) = $f(a.i, b.i)
end

(::Type{T})(x::EndIndex) where {T<:Number} = T(x.i)

Base.show(io::IO, x::EndIndex) = show(io, x.i)
Base.convert(::Type{T}, x::EndIndex) where {T<:Number} = convert(T, x.i)
function Base.promote_rule(::Type{EndIndex}, ::Type{T}) where {T<:Number}
    return promote_type(Int, T) == Int ? EndIndex : T
end
