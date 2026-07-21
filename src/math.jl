# Unary operators.
for (f, op_type) in [
    (:acosh, "Acosh")
    (:asin, "Asin")
    (:asinh, "Asinh")
    (:atan, "Atan") # Atan2 is not a supported ONNX operator.
    (:atanh, "Atanh")
    (:cos, "Cos")
    (:cosh, "Cosh")
    (:exp, "Exp")
    (:log, "Log")
    (:inv, "Reciprocal")
    (:sin, "Sin")
    (:sinh, "Sinh")
    (:sqrt, "Sqrt")
    (:tan, "Tan")
    (:tanh, "Tanh")
]
    @eval begin
        function Base.$f(A::AbstractProbeNumber)
            return onnx_op($op_type, float(eltype(A))(A))
        end
        function Base.$f(A::AbstractProbeNumber{<:AbstractFloat})
            return onnx_op($op_type, A)
        end
    end
end

for (f, op_type) in [
    (:abs, "Abs")
    (:-, "Neg")
    (:sign, "Sign")
]
    @eval begin
        function Base.$f(A::AbstractProbeNumber)
            return onnx_op($op_type, A)
        end
    end
end

function Base.:~(A::AbstractProbeNumber{<:Integer})
    return onnx_op("BitwiseNot", A)
end
function Base.:!(A::AbstractProbeNumber{Bool})
    return onnx_op("Not", A)
end

for (f, op_type) in [
    (:ceil, "Ceil")
    (:floor, "Floor")
    (:round, "Round")
]
    @eval begin
        Base.$f(A::AbstractProbeNumber{<:Integer}) = A
        function Base.$f(A::AbstractProbeNumber)
            return onnx_op($op_type, A)
        end
        function Base.$f(::Type{T}, A::ProbeNumber) where {T}
            return convert(ProbeNumber{T}, $f(A))
        end
        function Base.$f(::Type{T}, A::BroadcastProbe) where {T}
            return convert(BroadcastProbe{T}, $f(A))
        end
    end
end

for (f, op_type) in [
    (:isinf, "IsInf")
    (:isnan, "IsNan")
]
    @eval begin
        function Base.$f(A::AbstractProbeNumber{<:AbstractFloat})
            return onnx_op($op_type, Bool, A)
        end
        function Base.$f(A::AbstractProbeNumber{<:Complex{<:AbstractFloat}})
            return onnx_op($op_type, Bool, A)
        end
        function Base.$f(A::AbstractProbeNumber)
            Ap = A
            if raw_size(Ap) isa Dims
                return falses(raw_size(Ap))
            end
            shape = onnx_op("Shape", Int64, (ndims(A),), Ap)
            return onnx_op("ConstantOfShape", shape; attr=(value=TensorProto([false]),))
        end
    end
end

# Binary operators.
for (f, op_type) in [
    (:+, "Add")
    (:/, "Div")
    (:*, "Mul")
    (:-, "Sub")
]
    @eval begin
        function Base.$f(A::AbstractProbeNumber{T}, B::AbstractProbeNumber{T}) where {T}
            return onnx_op($op_type, A, B)
        end
    end
end

for (f, op_type) in [
    (:max, "Max")
    (:min, "Min")
    (:mod, "Mod")
]
    @eval begin
        Base.$f(A::Number, B::AbstractProbeNumber) = $f(promote(A, B)...)
        Base.$f(A::AbstractProbeNumber, B::Number) = $f(promote(A, B)...)
        function Base.$f(A::AbstractProbeNumber, B::AbstractProbeNumber)
            @assert eltype(A) != eltype(B) "Wrong method was dispatched"
            return $f(promote(A, B)...)
        end
        function Base.$f(A::AbstractProbeNumber{T}, B::AbstractProbeNumber{T}) where {T}
            return onnx_op($op_type, A, B)
        end
    end
end
function Base.mod(
    A::AbstractProbeNumber{T}, B::AbstractProbeNumber{T}
) where {T<:AbstractFloat}
    return onnx_op("Mod", A, B; attr=(fmod=1,))
end

# There are fallback implementations for nand and nor.
for (f, op_type) in [
    (:&, "BitwiseAnd")
    (:|, "BitwiseOr")
    (:xor, "BitwiseXor")
]
    @eval begin
        Base.$f(A::Integer, B::AbstractProbeNumber{<:Integer}) = $f(promote(A, B)...)
        Base.$f(A::AbstractProbeNumber{<:Integer}, B::Integer) = $f(promote(A, B)...)
        function Base.$f(
            A::AbstractProbeNumber{<:Integer}, B::AbstractProbeNumber{<:Integer}
        )
            @assert eltype(A) != eltype(B) "Wrong method was dispatched"
            return $f(promote(A, B)...)
        end
        function Base.$f(
            A::AbstractProbeNumber{T}, B::AbstractProbeNumber{T}
        ) where {T<:Integer}
            return onnx_op($op_type, A, B)
        end
    end
end

function Base.:+(
    A::AbstractProbeNumber{T},
    B::AbstractProbeNumber{T},
    C::AbstractProbeNumber{T},
    D::AbstractProbeNumber{T}...,
) where {T<:AbstractFloat}
    return onnx_op("Sum", A, B, C, D...)
end

# To prevent fallback implementation using repeated multiplications.
Base.:^(A::AbstractProbeNumber, B::Integer) = ^(A, wrap_broadcast(probe(B)))
Base.:^(A::AbstractProbeNumber, B::Number) = ^(A, wrap_broadcast(probe(B)))
function Base.:^(A::AbstractProbeNumber, B::AbstractProbeNumber)
    # Pow does not require arguments of the same type.
    return onnx_op("Pow", A, B)
end

# ONNX only support bit shift of unsigned integers.
function Base.:<<(A::AbstractProbeNumber{<:Unsigned}, B::Signed)
    return B < 0 ? A >> unsigned(-B) : A << unsigned(B)
end
function Base.:<<(A::AbstractProbeNumber{T}, B::Unsigned) where {T<:Unsigned}
    return iszero(one(T) << B) ? zero(T) : A << probe(T(B))
end
function Base.:<<(A::AbstractProbeNumber{T}, B::AbstractProbeNumber{T}) where {T<:Unsigned}
    return onnx_op("BitShift", A, B; attr=(direction="LEFT",))
end

Base.:>>>(A::AbstractProbeNumber{<:Unsigned}, B::Number) = A >> B
function Base.:>>(A::AbstractProbeNumber{<:Unsigned}, B::Signed)
    return B < 0 ? A << unsigned(-B) : A >> unsigned(B)
end
function Base.:>>(A::AbstractProbeNumber{T}, B::Unsigned) where {T<:Unsigned}
    return iszero(typemax(T) >> B) ? zero(T) : A >> probe(T(B))
end
function Base.:>>(A::AbstractProbeNumber{T}, B::AbstractProbeNumber{T}) where {T<:Unsigned}
    return onnx_op("BitShift", A, B; attr=(direction="RIGHT",))
end
# TODO: Add support for shifting constant, e.g., 2 << B.

# There is a fallback implementation !=(x, y) = !(x == y).
for (f, op_type) in [
    (:(==), "Equal")
    (:>, "Greater")
    (:>=, "GreaterOrEqual")
    (:<, "Less")
    (:<=, "LessOrEqual")
]
    @eval begin
        # There is no automatic promotion for comparisons.
        Base.$f(A::AbstractProbeNumber, B::Number) = $f(promote(A, B)...)
        Base.$f(A::Number, B::AbstractProbeNumber) = $f(promote(A, B)...)
        function Base.$f(A::AbstractProbeNumber, B::AbstractProbeNumber)
            # Assert to avoid infinite recursion.
            @assert typeof(A) != typeof(B) "Wrong method was dispatched"
            return $f(promote(A, B)...)
        end
        function Base.$f(A::AbstractProbeNumber{T}, B::AbstractProbeNumber{T}) where {T}
            return onnx_op($op_type, Bool, A, B)
        end
    end
end

# There are fallback implementations for nand and nor.
for (f, op_type) in [
    (:&, "And")
    (:|, "Or")
    (:xor, "Xor")
]
    @eval begin
        function Base.$f(A::AbstractProbeNumber{Bool}, B::AbstractProbeNumber{Bool})
            return onnx_op($op_type, A, B)
        end
    end
end

@overload Base.clamp _clamp AbstractProbeNumber Any 3
_clamp(x, lo, hi) = clamp(promote(x, lo, hi)...)
function Base.clamp(
    x::AbstractProbeNumber{T}, lo::AbstractProbeNumber{T}, hi::AbstractProbeNumber{T}
) where {T}
    return onnx_op("Clip", x, lo, hi)
end
Base.clamp(x::AbstractProbeNumber{T}, ::Type{T}) where {T} = x
function Base.clamp(x::AbstractProbeNumber{S}, ::Type{T}) where {S,T}
    y = clamp(x, typemin(T), typemax(T))
    return convert(AbstractProbeNumber{T}, y)
end
