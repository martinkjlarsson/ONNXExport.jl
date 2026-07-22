function load(file_name::String)
    return open(file_name, "r") do io
        return load(io)
    end
end

function load(io::IO)
    return decode(ProtoDecoder(io), ModelProto)
end

function save(file_name::String, model::ModelProto)
    return open(file_name, "w") do io
        return save(io, model)
    end
end

function save(io::IO, model::ModelProto)
    return encode(ProtoEncoder(io), model)
end
