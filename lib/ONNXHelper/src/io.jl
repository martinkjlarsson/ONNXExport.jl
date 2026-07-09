function loadmodel(file_name::String)
    return open(file_name, "r") do io
        loadmodel(io)
    end
end

function loadmodel(io::IO)
    return decode(ProtoDecoder(io), ModelProto)
end

function savemodel(file_name::String, model::ModelProto)
    return open(file_name, "w") do io
        savemodel(io, model)
    end
end

function savemodel(io::IO, model::ModelProto)
    return encode(ProtoEncoder(io), model)
end
