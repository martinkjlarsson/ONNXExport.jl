function NodeProto(
    op_type::String,
    input,
    output,
    attribute::AbstractVector{AttributeProto}=AttributeProto[];
    domain="",
    overload="",
    name="",
    doc_string="",
    metadata_props=StringStringEntryProto[],
    device_configurations=NodeDeviceConfigurationProto[],
)
    return NodeProto(
        input,
        output,
        name,
        op_type,
        domain,
        overload,
        attribute,
        doc_string,
        metadata_props,
        device_configurations,
    )
end

function NodeProto(op_type::String, input, output, attribute; kwargs...)
    return NodeProto(
        op_type,
        input,
        output,
        AttributeProto[AttributeProto(string(k), v) for (k, v) in pairs(attribute)];
        kwargs...,
    )
end
