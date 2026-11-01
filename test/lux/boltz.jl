using Boltz, JLD2, Lux, Metalhead, MLUtils, ONNXExport, ONNXRunTime, Pickle, Random, Test

include("../utils.jl")

function save_model(io, model, x, ps, st::NamedTuple; kwargs...)
    f(x) = first(Lux.apply(model, x, ps, st))
    return ONNXExport.save(io, f, x; kwargs...)
end
function save_model(io, model, x::Tuple, ps, st::NamedTuple; kwargs...)
    f(x...) = first(Lux.apply(model, x, ps, st))
    return ONNXExport.save(io, f, x...; kwargs...)
end

function test_boltz_model(model; input_size=(224, 244, 3), batch_size=4)
    rng = Random.default_rng()
    x = rand(Float32, input_size..., batch_size)
    y, y_onnx = test_model(rng, model, x; test=true, opset_version=23)

    return y, y_onnx
end

# Needed for VisionTransformer.
function Boltz.Layers.ones_batch_like(x::ProbeArray{T,N}) where {T,N}
    return MLUtils.ones_like(x, (ntuple(_ -> 1, N - 1)..., size(x, N)))
end

pretrained = false

@testset "Computer Vision Models" begin
    # Native Lux Models.
    @testset "AlexNet" begin
        model = Vision.AlexNet(; pretrained="ImageNet1K")
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    @testset "EfficientNet" begin
        model = Vision.EfficientNet(:b0; pretrained=pretrained)
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    @testset "VGG" begin
        model = Vision.VGG(11; pretrained=pretrained)
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    @testset "VisionTransformer" begin
        # Pretrained not implemented.
        model = Vision.VisionTransformer(:tiny; pretrained=false)
        y, y_onnx = test_boltz_model(model; input_size=(256, 256, 3))
        @test y_onnx ≈ y
    end

    # Imported from Metalhead.jl.
    @testset "ConvMixer" begin
        model = Vision.ConvMixer(:base)
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    @testset "DenseNet" begin
        model = Vision.DenseNet(121)
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    @testset "GoogLeNet" begin
        model = Vision.GoogLeNet()
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    # Uncomment when Metalhead has been updated.
    # @testset "MobileNet" begin
    #     model = Vision.MobileNet(:v3_small)
    #     y, y_onnx = test_batched_model(model)
    #     @test y_onnx ≈ y
    # end

    @testset "ResNet" begin
        model = Vision.ResNet(18; pretrained=pretrained)
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    @testset "ResNeXt" begin
        model = Vision.ResNeXt(50; pretrained=pretrained)
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    @testset "SqueezeNet" begin
        model = Vision.SqueezeNet(; pretrained=pretrained)
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end

    @testset "WideResNet" begin
        model = Vision.WideResNet(50; pretrained=pretrained)
        y, y_onnx = test_boltz_model(model)
        @test y_onnx ≈ y
    end
end

@testset "Physics-Informed Machine Learning (PIML)." begin
    @testset "Transolver" begin
        model = PIML.Transolver(;
            func_dim=8, spatial_dim=2, nheads=8, num_layers=1, out_dim=2, use_rms_norm=true
        )
        y, y_onnx = test_boltz_model(model; input_size=(5, 32))
        @test y_onnx ≈ y
    end
end
