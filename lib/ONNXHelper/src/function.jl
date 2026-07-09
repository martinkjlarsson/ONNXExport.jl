# TODO: Which arguments are mandatory and which should be keyword arguments?
function FunctionProto(
    domain::String,
    name::String,
    input,
    output,
    node,
    attribute=String[];
    doc_string="",
    opset_import=OperatorSetIdProto[],
    overload="",
    value_info=ValueInfoProto[],
    metadata_props=StringStringEntryProto[],
)
    as = filter(a -> a isa String, attribute)
    ap = filter(a -> a isa AttributeProto, attribute)

    return FunctionProto(
        name,
        input,
        output,
        as,
        ap,
        node,
        doc_string,
        opset_import,
        domain,
        overload,
        value_info,
        metadata_props,
    )
end
