# ONNXHelper.jl

`ONNXHelper.jl` is a Julia package providing convenience methods for working with [ONNX](https://onnx.ai/) files. Support for `bfloat16` and the 8-bit floats `float8e4m3fn`, `float8e5m2`, and `float8e8m0` are provided through extensions using [BFloat16s.jl](https://github.com/JuliaMath/BFloat16s.jl) and [Microfloats.jl](https://github.com/MurrellGroup/Microfloats.jl), respectively.

# Other ONNX packages
* [ONNX.jl](https://github.com/FluxML/ONNX.jl) supports saving, loading, and executing ONNX graphs.
* [ONNXLowLevel.jl](https://github.com/GunnarFarneback/ONNXLowLevel.jl) provides a similar functionality to this package.
* [ONNXNaiveNASflux.jl](https://github.com/DrChainsaw/ONNXNaiveNASflux.jl) allows for import and export of [Flux.jl](https://github.com/FluxML/Flux.jl) models.
* [ONNXRunTime.jl](https://github.com/jw3126/ONNXRunTime.jl) supports loading and running ONNX graph by providing bindings to the ONNX Runtime C API.
