function load_model(file_name::String)
    return open(file_name, "r") do io
        load_model(io)
    end
end

function load_model(io::IO)
    return decode(ProtoDecoder(io), ModelProto)
end

function save_model(file_name::String, model::ModelProto)
    return open(file_name, "w") do io
        save_model(io, model)
    end
end

function save_model(io::IO, model::ModelProto)
    return encode(ProtoEncoder(io), model)
end
