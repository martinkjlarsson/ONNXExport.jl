"""
    ONNXExport.stft(x::AbstractArray{T,3}; kwargs...) where {T<:Real}

Short-time Fourier transform (STFT).

Wrapper for `NNlib.stft` that uses a real interleaved representation of complex numbers
instead of the `Complex` number type.
`x` must be of shape `(1, L, B)` for real inputs and `(2, L, B)` for complex inputs, where
`x[1, :, :]` and `x[2, :, :]` (when provided) contain the real and imaginary components,
respectively.
Similarly, the returned result `y` is of size `(2, n_fft, n_frames, B)` with real
`y[1, :, :, :]` and imaginary `y[2, :, :, :]` components.
The keyword arguments are the same as for `NNlib.stft`.

This functions exists to enable ONNX export of STFT. The STFT ONNX operator does not support
complex number and uses the interleaved representation described above instead.
Because of this, direct export of `NNlib.stft` is not supported.

See also [`NNlib.stft`](@ref).
"""
function stft end
