function NNlib.hamming_window(
    window_length::ProbeNumber{Int},
    ::Type{T}=Float32;
    periodic::Bool=true,
    α::T=T(0.54),
    β::T=T(0.46),
) where {T<:Real}
    new_dims = (ONNXExport.dimension_name(),)
    attr=(output_datatype=Int(tensor_type(T)), periodic=periodic)
    y = onnx_op("HammingWindow", T, new_dims, window_length; attr=attr)

    onnx_α = T(0.54347825f0) # == 25f0 / 46f0
    onnx_β = T(0.45652175f0) # == 21f0 / 46f0 == 1f0 - 25f0 / 46f0

    α==onnx_α && β==onnx_β && return y

    @warn "The ONNX operator HammingWindow uses α=0.54347825 and β=0.45652175. The " *
        "current values (α=$α, β=$β) require rescaling and extra ONNX operations. " *
        "Consider using the values from ONNX to avoid this."
    return y = y .* (β / onnx_β) .- (onnx_α * β / onnx_β - α)
end

function NNlib.hann_window(
    window_length::ProbeNumber{Int}, ::Type{T}=Float32; periodic::Bool=true
) where {T<:Real}
    new_dims = (ONNXExport.dimension_name(),)
    attr=(output_datatype=Int(tensor_type(T)), periodic=periodic)
    return onnx_op("HannWindow", T, new_dims, window_length; attr=attr)
end

function NNlib.stft(::ProbeArray; kwargs...)
    return error(
        "ONNX export of NNlib.stft is not directly supported as the corresponding ONNX " *
        "operator STFT does not support complex numbers. Use ONNXExport.stft instead.",
    )
end

function NNlib.istft(::ProbeArray; kwargs...)
    return error(
        "ONNX export of NNlib.istft is not supported as there is no corresponding ONNX " *
        "operator.",
    )
end

function ONNXExport.stft(x::AbstractArray{T,3}; kwargs...) where {T<:Real}
    if size(x, 1) == 1
        x1 = dropdims(x; dims=1)
    elseif size(x, 1) == 2
        x1 = reinterpret(reshape, Complex{T}, x)
    else
        throw(
            ArgumentError(
                "The first dimension of `x` must be either 1 for real inputs or 2 for " *
                "complex inputs. See the docstring for details",
            ),
        )
    end
    y = NNlib.stft(x1; kwargs...)
    return reinterpret(reshape, T, y)
end

function ONNXExport.stft(
    x::ProbeArray{T,3};
    n_fft::Union{Int,ProbeNumber{Int}},
    hop_length::Union{Int,ProbeNumber{Int}}=n_fft ÷ 4,
    window=nothing,
    center::Bool=true,
    normalized::Bool=false,
) where {T<:Real}
    @assert raw_size(x, 1) == 1 || raw_size(x, 1) == 2

    real_input = raw_size(x, 1) == 1

    if !real_input && !isnothing(window)
        @warn "The STFT operator might not yield correct outputs for complex inputs when " *
            "a window is provided. See the ORT issue " *
            "https://github.com/microsoft/onnxruntime/issues/33132."
    end

    if center
        pad_amount = n_fft ÷ 2
        x = pad_reflect(x, pad_amount; dims=2)
    end

    n = raw_size(x, 2)
    if n isa Symbol || isprobe(n_fft) || isprobe(hop_length)
        n_frames = ONNXExport.dimension_name()
    else
        n_frames = 1 + (n - n_fft) ÷ hop_length
    end
    if isprobe(n_fft)
        dft_unique_bins = ONNXExport.dimension_name()
    else
        dft_unique_bins = real_input ? n_fft÷2 + 1 : n_fft
    end
    new_dims = (2, dft_unique_bins, n_frames, raw_size(x, 3))

    frame_step = probe(hop_length, "frame_step")
    window = probe(window, "window")
    frame_length = probe(n_fft)
    attr = (onesided=real_input,)

    y = onnx_op("STFT", new_dims, x, frame_step, window, frame_length; attr=attr)

    if normalized
        y = y .* T(n_fft)^T(-0.5)
    end

    return y
end
