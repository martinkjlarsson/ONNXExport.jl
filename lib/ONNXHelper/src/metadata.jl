# TODO: Apply everywhere the metadata_props argument is used.
convert_metadata(md::Vector{StringStringEntryProto}) = md
convert_metadata(md::Dict{String,String}) = [StringStringEntryProto(k, v) for (k, v) in md]
convert_metadata(md::NamedTuple) = [StringStringEntryProto(k, v) for (k, v) in pairs(md)]
