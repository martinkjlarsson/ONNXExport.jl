"""
    tensor_type(type)

Determine the `var"TensorProto.DataType"` corresponding to the given `type`. Types with no
correspondence will throw an error.
The definition `tensor_type(x) = tensor_type(typeof(x))` is provided for convenience so
that instances can be passed instead of types.
"""
tensor_type(x) = tensor_type(typeof(x))
tensor_type(::Type{T}) where {T} = error("$T does not map to an ONNX tensor data type.")
tensor_type(::Type{Float32}) = var"TensorProto.DataType".FLOAT
tensor_type(::Type{UInt8}) = var"TensorProto.DataType".UINT8
tensor_type(::Type{Int8}) = var"TensorProto.DataType".INT8
tensor_type(::Type{UInt16}) = var"TensorProto.DataType".UINT16
tensor_type(::Type{Int16}) = var"TensorProto.DataType".INT16
tensor_type(::Type{Int32}) = var"TensorProto.DataType".INT32
tensor_type(::Type{Int64}) = var"TensorProto.DataType".INT64
tensor_type(::Type{<:AbstractString}) = var"TensorProto.DataType".STRING
tensor_type(::Type{Bool}) = var"TensorProto.DataType".BOOL
tensor_type(::Type{Float16}) = var"TensorProto.DataType".FLOAT16
tensor_type(::Type{Float64}) = var"TensorProto.DataType".DOUBLE
tensor_type(::Type{UInt32}) = var"TensorProto.DataType".UINT32
tensor_type(::Type{UInt64}) = var"TensorProto.DataType".UINT64
tensor_type(::Type{ComplexF32}) = var"TensorProto.DataType".COMPLEX64
tensor_type(::Type{ComplexF64}) = var"TensorProto.DataType".COMPLEX128
# tensor_type(::Type{}) = var"TensorProto.DataType".FLOAT8E4M3FN
# tensor_type(::Type{}) = var"TensorProto.DataType".FLOAT8E4M3FNUZ
# tensor_type(::Type{}) = var"TensorProto.DataType".FLOAT8E5M2
# tensor_type(::Type{}) = var"TensorProto.DataType".FLOAT8E5M2FNUZ
# tensor_type(::Type{}) = var"TensorProto.DataType".UINT4
# tensor_type(::Type{}) = var"TensorProto.DataType".INT4
# tensor_type(::Type{}) = var"TensorProto.DataType".FLOAT4E2M1
# tensor_type(::Type{}) = var"TensorProto.DataType".FLOAT8E8M0

"""
    julia_type(data_type::var"TensorProto.DataType".T)

Determine the Julia type corresponding to the given `var"TensorProto.DataType"`
`data_type`. Data types with no correspondence will throw an error.
"""
julia_type(data_type::var"TensorProto.DataType".T) = julia_type(Val(data_type))
julia_type(::Val{DT}) where {DT} = julia_type_error(DT)
julia_type(::Val{var"TensorProto.DataType".FLOAT}) = Float32
julia_type(::Val{var"TensorProto.DataType".UINT8}) = UInt8
julia_type(::Val{var"TensorProto.DataType".INT8}) = Int8
julia_type(::Val{var"TensorProto.DataType".UINT16}) = UInt16
julia_type(::Val{var"TensorProto.DataType".INT16}) = Int16
julia_type(::Val{var"TensorProto.DataType".INT32}) = Int32
julia_type(::Val{var"TensorProto.DataType".INT64}) = Int64
julia_type(::Val{var"TensorProto.DataType".STRING}) = String
julia_type(::Val{var"TensorProto.DataType".BOOL}) = Bool
julia_type(::Val{var"TensorProto.DataType".FLOAT16}) = Float16
julia_type(::Val{var"TensorProto.DataType".DOUBLE}) = Float64
julia_type(::Val{var"TensorProto.DataType".UINT32}) = UInt32
julia_type(::Val{var"TensorProto.DataType".UINT64}) = UInt64
julia_type(::Val{var"TensorProto.DataType".COMPLEX64}) = ComplexF32
julia_type(::Val{var"TensorProto.DataType".COMPLEX128}) = ComplexF64

function julia_type_error(data_type::var"TensorProto.DataType".T)
    if data_type == var"TensorProto.DataType".BFLOAT16
        error(
            "The bfloat16 tensor data type is available through the BFloat16s package " *
            "extension. Add \"using BFloat16s\" to your source code.",
        )
    elseif data_type ∈ (
        var"TensorProto.DataType".FLOAT8E4M3FN,
        var"TensorProto.DataType".FLOAT8E5M2,
        var"TensorProto.DataType".FLOAT8E8M0,
    )
        error(
            "The float8e4m3fn, float8e5m2, and float8e8m0 tensor data types are " *
            "available through the Microfloats package extension. Add " *
            "\"using Microfloats\" to your source code.",
        )
    else
        error("The tensor data type ", data_type, " does not map to a Julia type")
    end
end

"""
    tensor_to_field(array::AbstractArray)

Convert an `array` to a type suitable for the corresponding proto field in TensorProto.
"""
tensor_to_field(array::AbstractArray) = vec(array)
tensor_to_field(array::AbstractArray{<:AbstractString}) = codeunits.(array)
tensor_to_field(array::AbstractArray{Float16}) = reinterpret(UInt16, vec(array))
tensor_to_field(array::AbstractArray{ComplexF32}) = reinterpret(Float32, vec(array))
tensor_to_field(array::AbstractArray{ComplexF64}) = reinterpret(Float64, vec(array))

"""
    TensorProto(array::AbstractArray{T}; raw=false, name="", doc_string="", metadata_props=[])

Create a `TensorProto` containing the provided `array`.

`T` must be a type for which `tensor_type(T)` returns a valid `var"TensorProto.DataType"`.
If `raw=false`, one of the dedicated proto fields are used to store the array, while if
`raw=true`, the `raw_data` proto field is used. `raw=false` may result in smaller file
sizes for large integer types due to protobuf's
[variable-width integer](https://protobuf.dev/programming-guides/encoding/#varints)
encoding. However, `raw=true` may result in faster loading in general and smaller file
sizes for small data types. Raw encoding offers no benefit for Float32 or Float64 data and
is not supported for strings.
"""
function TensorProto(
    array::AbstractArray{T}; raw=false, kwargs...
) where {T<:Union{Number,AbstractString}}
    if raw
        return _from_array_raw(array; kwargs...)
    else
        return _from_array(Val(tensor_type(T)), array; kwargs...)
    end
end

"""
    TensorProto(scalar::T; kwargs...)

Create a `TensorProto` containing the provided `scalar`.

This is equivalent to `TensorProto(fill(scalar))`. See the array version for explanation of
the keyword arguments.
"""
function TensorProto(scalar::T; kwargs...) where {T<:Union{Number,AbstractString}}
    return TensorProto(fill(scalar); kwargs...)
end

# TODO: Constructors for external data.

function _from_array_raw(array::AbstractArray{T}; kwargs...) where {T}
    if T <: AbstractString
        throw(ArgumentError("Raw serialization is not supported for strings."))
    end
    if ENDIAN_BOM == 0x01020304 # Big-endian machine.
        array = htol.(array)
    end

    return _tensorproto(;
        dims=collect(reverse(size(array))),
        data_type=Int32(tensor_type(T)),
        raw_data=reinterpret(UInt8, vec(array)),
        kwargs...,
    )
end

for (DT, F) in [
    (var"TensorProto.DataType".FLOAT, :float_data)
    (var"TensorProto.DataType".UINT8, :int32_data)
    (var"TensorProto.DataType".INT8, :int32_data)
    (var"TensorProto.DataType".UINT16, :int32_data)
    (var"TensorProto.DataType".INT16, :int32_data)
    (var"TensorProto.DataType".INT32, :int32_data)
    (var"TensorProto.DataType".INT64, :int64_data)
    (var"TensorProto.DataType".STRING, :string_data)
    (var"TensorProto.DataType".BOOL, :int32_data)
    (var"TensorProto.DataType".FLOAT16, :int32_data)
    (var"TensorProto.DataType".DOUBLE, :double_data)
    (var"TensorProto.DataType".UINT32, :uint64_data)
    (var"TensorProto.DataType".UINT64, :uint64_data)
    (var"TensorProto.DataType".COMPLEX64, :float_data)
    (var"TensorProto.DataType".COMPLEX128, :double_data)
    (var"TensorProto.DataType".BFLOAT16, :int32_data)
    (var"TensorProto.DataType".FLOAT8E4M3FN, :int32_data)
    (var"TensorProto.DataType".FLOAT8E4M3FNUZ, :int32_data)
    (var"TensorProto.DataType".FLOAT8E5M2, :int32_data)
    (var"TensorProto.DataType".FLOAT8E5M2FNUZ, :int32_data)
    (var"TensorProto.DataType".UINT4, :int32_data)
    (var"TensorProto.DataType".INT4, :int32_data)
    (var"TensorProto.DataType".FLOAT4E2M1, :int32_data)
    (var"TensorProto.DataType".FLOAT8E8M0, :int32_data)
]
    @eval begin
        function _from_array(
            data_type::Val{$DT}, array::AbstractArray{T}; kwargs...
        ) where {T}
            @assert T <: julia_type(data_type) "Array type does not match tensor data type"
            return _tensorproto(;
                dims=collect(reverse(size(array))),
                data_type=Int32($DT),
                $F=tensor_to_field(array),
                kwargs...,
            )
        end
        function _get_data(::Val{$DT}, tensor::TensorProto)
            return tensor.$F
        end
    end
end

function _tensorproto(;
    dims=Vector{Int64}(),
    data_type=zero(Int32),
    segment=nothing,
    float_data=Vector{Float32}(),
    int32_data=Vector{Int32}(),
    string_data=Vector{Vector{UInt8}}(),
    int64_data=Vector{Int64}(),
    name="",
    doc_string="",
    raw_data=UInt8[],
    external_data=Vector{StringStringEntryProto}(),
    data_location=var"TensorProto.DataLocation".DEFAULT,
    double_data=Vector{Float64}(),
    uint64_data=Vector{UInt64}(),
    metadata_props=Vector{StringStringEntryProto}(),
)
    return TensorProto(
        dims,
        data_type,
        segment,
        float_data,
        int32_data,
        string_data,
        int64_data,
        name,
        doc_string,
        raw_data,
        external_data,
        data_location,
        double_data,
        uint64_data,
        metadata_props,
    )
end

"""
    get_data(tensor::TensorProto)

Get the data field for the given tensor.
"""
function get_data(tensor::TensorProto)
    data_type = var"TensorProto.DataType".T(tensor.data_type)
    return _get_data(Val(data_type), tensor)
end

"""
    to_array(tensor::TensorProto)

Convert a `TensorProto` to a Julia `Array` of appropriate element type. The returned array
may or may not share the same underlying data as the tensor.
"""
function to_array(tensor::TensorProto)
    data_type = var"TensorProto.DataType".T(tensor.data_type)
    T = julia_type(data_type) # Placed here so data_type is validated.
    if isempty(tensor.raw_data)
        return to_array_typed(Val(data_type), tensor)
    else
        vec_data = reinterpret(T, tensor.raw_data)
        return reshape(vec_data, reverse(tensor.dims)...)
    end
end

for DT in [
    var"TensorProto.DataType".FLOAT,
    var"TensorProto.DataType".UINT8,
    var"TensorProto.DataType".INT8,
    var"TensorProto.DataType".UINT16,
    var"TensorProto.DataType".INT16,
    var"TensorProto.DataType".INT32,
    var"TensorProto.DataType".INT64,
    var"TensorProto.DataType".STRING,
    var"TensorProto.DataType".BOOL,
    var"TensorProto.DataType".FLOAT16,
    var"TensorProto.DataType".DOUBLE,
    var"TensorProto.DataType".UINT32,
    var"TensorProto.DataType".UINT64,
]
    T = julia_type(Val(DT))
    @eval begin
        function to_array_typed(data_type::Val{$DT}, tensor::TensorProto)
            vec_data = $T.(_get_data(data_type, tensor))
            return reshape(vec_data, reverse(tensor.dims)...)
        end
    end
end
function to_array_typed(::Val{var"TensorProto.DataType".COMPLEX64}, tensor::TensorProto)
    vec_data = reinterpret(ComplexF32, tensor.float_data)
    return reshape(vec_data, reverse(tensor.dims)...)
end
function to_array_typed(::Val{var"TensorProto.DataType".COMPLEX128}, tensor::TensorProto)
    vec_data = reinterpret(ComplexF64, tensor.double_data)
    return reshape(vec_data, reverse(tensor.dims)...)
end

function Base.show(io::IO, tensor::TensorProto)
    array = to_array(tensor)
    print(io, TensorProto, '(', array, "; raw=", !isempty(tensor.raw_data))
    tensor.name != "" && print(io, ", name=", tensor.name)
    tensor.doc_string != "" && print(io, ", doc_string=", tensor.doc_string)
    if !isempty(tensor.metadata_props)
        print(io, ", metadata_props=", tensor.metadata_props)
    end
    print(io, ')')
    return nothing
end
