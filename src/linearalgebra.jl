# TODO: We could support Diagonal{T, ProbeVector{T}}, but it raises other issues.
Base.:*(x::ProbeNumber, D::Diagonal) = x * probe(D)
Base.:*(D::Diagonal, x::ProbeNumber) = probe(D) * x

# TODO: Implement version for mapslices with dims=(1,2).
function LinearAlgebra.det(A::ProbeMatrix{T}) where {T<:Real}
    return det(convert(ProbeMatrix{float(T)}, A))
end
function LinearAlgebra.det(A::ProbeMatrix{T}) where {T<:AbstractFloat}
    return onnx_op("Det", (), A)
end
LinearAlgebra.logdet(A::ProbeMatrix) = log(det(A)) # TODO: Is it better to not define these?
function LinearAlgebra.logabsdet(A::ProbeMatrix)
    d = det(A)
    return (log(abs(d)), sign(d))
end

function LinearAlgebra.dot(x::ProbeArray{T,N}, y::ProbeArray{T,N}) where {T<:Real,N}
    return matmul_onnx(vec(x), vec(y))
end
function LinearAlgebra.dot(
    x::ProbeVector{T}, A::ProbeMatrix{T}, y::ProbeVector{T}
) where {T<:Real}
    return dot(x, A * y)
end

function (I::UniformScaling{T})(n::ProbeNumber{<:Integer}) where {T}
    return I(convert(ProbeNumber{Int64}, n))
end
function (I::UniformScaling{T})(n::ProbeNumber{Int64}) where {T}
    dim = dimension_name()
    shape = fill(n, 2)
    zeros = onnx_op(
        "ConstantOfShape", T, (dim, dim), shape; attr=(value=TensorProto([zero(T)]),)
    )
    if iszero(I.λ)
        return zeros
    end
    eye = onnx_op("EyeLike", zeros)
    if isone(I.λ)
        return eye
    end
    lambda = probe(I.λ, "lambda")
    return onnx_op("Mul", lambda, eye)
end

LinearAlgebra.issymmetric(::ProbeArray) = unsupported(issymmetric)
LinearAlgebra.isposdef(::ProbeArray) = unsupported(isposdef)
LinearAlgebra.isposdef!(::ProbeArray) = unsupported(isposdef!)
LinearAlgebra.istril(::ProbeArray) = unsupported(istril)
LinearAlgebra.istriu(::ProbeArray) = unsupported(istriu)
LinearAlgebra.isdiag(::ProbeArray) = unsupported(isdiag)
LinearAlgebra.ishermitian(::ProbeArray) = unsupported(ishermitian)

LinearAlgebra.tril(M::ProbeMatrix, k::Integer=0) = _trilu(M, Int64(k), true)
LinearAlgebra.tril(M::AbstractMatrix, k::ProbeNumber{<:Integer}) = _trilu(M, k, true)
LinearAlgebra.tril(M::ProbeMatrix, k::ProbeNumber{<:Integer}) = _trilu(M, k, true)

LinearAlgebra.triu(M::ProbeMatrix, k::Integer=0) = _trilu(M, Int64(k), false)
LinearAlgebra.triu(M::AbstractMatrix, k::ProbeNumber{<:Integer}) = _trilu(M, k, false)
LinearAlgebra.triu(M::ProbeMatrix, k::ProbeNumber{<:Integer}) = _trilu(M, k, false)

# Due to row-major order, we must negate k and upper/lower.
function _trilu(M::ProbeMatrix, k::Int64, lower)
    k = probe(-k, "k")
    return onnx_op("Trilu", M, k; attr=(upper=Int64(lower),))
end
function _trilu(M, k::ProbeNumber, lower)
    M = probe(M)
    k = convert(ProbeNumber{Int64}, k)
    return onnx_op("Trilu", M, -k; attr=(upper=Int64(lower),))
end

function LinearAlgebra.norm(A::ProbeArray, p::Real=2)
    A = convert(ProbeArray{float(eltype(A))}, A)
    if p == 2
        return _reduce("ReduceL2", A, :)
    elseif p == 1
        return _reduce("ReduceL1", A, :)
    elseif p == Inf
        return _reduce("ReduceMax", A, :)
    elseif p == 0
        return convert(ProbeNumber{eltype(A)}, count(!iszero, A))
    elseif p == -Inf
        return _reduce("ReduceMin", A, :)
    else
        return sum(abs.(A) .^ p)^(1 / p)
    end
end

function LinearAlgebra.opnorm(A::ProbeMatrix, p::Real=2)
    A = convert(ProbeArray{float(eltype(A))}, A)
    if p == 2
        error("opnorm with p=2 is not supported for ONNX export")
    elseif p == 1
        return _reduce("ReduceMax", _reduce("ReduceL1", A, 1), :)
    elseif p == Inf
        return _reduce("ReduceMax", _reduce("ReduceL1", A, 2), :)
    else
        throw(ArgumentError(lazy"invalid p-norm p=$p. Valid: 1, 2, Inf"))
    end
end

LinearAlgebra.normalize(A::ProbeArray, p::Real=2) = A ./ norm(A, p)
function LinearAlgebra.normalize(A::ProbeVector, p::Real=2)
    A = convert(ProbeArray{float(eltype(A))}, A)
    if p == 2
        return onnx_op("LpNormalization", A)
    elseif p == 1
        return onnx_op("LpNormalization", A; attr=(p=1,))
    else
        return A ./ norm(A, p)
    end
end

LinearAlgebra.kron(A::ProbeVecOrMat, B::ProbeVecOrMat) = kron(promote(A, B)...)
function LinearAlgebra.kron(A::ProbeVector{T}, B::ProbeVector{T}) where {T}
    return vec(B .* transpose(A))
end
function LinearAlgebra.kron(A::ProbeVecOrMat{T}, B::ProbeVecOrMat{T}) where {T}
    A2 = unsqueeze(A, (1, 3))
    B2 = ndims(B) == 1 ? B : unsqueeze(B, (2, 4))

    C = A2 .* B2

    # Avoid unnecessary Shape calls.
    as1 = raw_size(A, 1)
    bs1 = raw_size(B, 1)

    if isa(as1, Int) && isa(bs1, Int)
        return reshape(C, as1 * bs1, :)
    else
        return reshape(C, :, size(A, 2) * size(B, 2))
    end
end
