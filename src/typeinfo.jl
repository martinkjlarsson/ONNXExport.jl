"""
    TypeInfo([name,] [T=Float32,] [dims::Tuple])
    TypeInfo([name,] [T=Float32,] dims...)

Construct an input template, indicating the name, type, and size of an ONNX input tensor.

If `name` is omitted, a generic name will be given, e.g., "input1". `T` must be a `Number`
type and defaults to `Float32`. If `dims` is omitted, the input is assumed to be a scalar,
otherwise the input corresponds to an array with the given dimensions.
"""
struct TypeInfo
    name::String
    T::Type{<:Number}
    dims::ProbeDims
    isscalar::Bool

    function TypeInfo(name::String, T::Type, dims::ProbeDims, isscalar::Bool)
        isscalar && !isempty(dims) && error("dims must be an empty tuple if isscalar=true")

        return new(name, T, dims, isscalar)
    end
end

TypeInfo(dims::ProbeDims) = TypeInfo("", Float32, dims, false)
TypeInfo(dims::ProbeDim...) = TypeInfo("", Float32, dims, isempty(dims))
TypeInfo(T::Type, dims::ProbeDims) = TypeInfo("", T, dims, false)
TypeInfo(T::Type, dims::ProbeDim...) = TypeInfo("", T, dims, isempty(dims))
TypeInfo(name::String, dims::ProbeDims) = TypeInfo(name, Float32, dims, false)
TypeInfo(name::String, dims::ProbeDim...) = TypeInfo(name, Float32, dims, isempty(dims))
TypeInfo(name::String, T::Type, dims::ProbeDims) = TypeInfo(name, T, dims, false)
TypeInfo(name::String, T::Type, dims::ProbeDim...) = TypeInfo(name, T, dims, isempty(dims))

name(ti::TypeInfo) = ti.name
Base.eltype(ti::TypeInfo) = ti.T
raw_size(ti::TypeInfo) = ti.dims
isscalar(ti::TypeInfo) = ti.isscalar

type_info(ti::TypeInfo) = ti
type_info(A::AbstractArray) = TypeInfo(eltype(A), size(A))
type_info(x::Number) = TypeInfo(typeof(x))
