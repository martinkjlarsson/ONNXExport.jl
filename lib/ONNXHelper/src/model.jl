function ModelProto(
    graph::GraphProto;
    ir_version=Version.IR_VERSION,
    opset_import=[OperatorSetIdProto("", 24)],
    producer_name=string(ONNXHelper) * ".jl",
    producer_version=string(pkgversion(ONNXHelper)),
    domain="",
    model_version=0,
    doc_string="",
    metadata_props=StringStringEntryProto[],
    training_info=TrainingInfoProto[],
    functions=FunctionProto[],
    configuration=DeviceConfigurationProto[],
)
    return ModelProto(
        Int64(ir_version),
        opset_import,
        producer_name,
        producer_version,
        domain,
        model_version,
        doc_string,
        graph,
        metadata_props,
        training_info,
        functions,
        configuration,
    )
end
