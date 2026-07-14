using Base.ScopedValues

"""
Represents an ONNX namespace, storing all used names for values, nodes, graphs, and
symbolic dimensions/shape variables. See the
[ONNX documentation](https://onnx.ai/onnx/repo-docs/IR.html#names-within-a-graph)
for details. For simplicity, values, nodes, and graphs share the same namespace.
"""
struct Namespace
    prefix::String
    names::Set{String}
    dimensions::Set{Symbol}
end
Namespace() = Namespace("", Set{String}(), Set{Symbol}())

const NAMESPACE = ScopedValue(Namespace())

get_value_name(name::String) = get_name(name)
get_node_name(name::String) = get_name(name)
get_graph_name(name::String) = get_name(name)

add_value(name::String) = add_name(name)
add_node(name::String) = add_name(name)
add_graph(name::String) = add_name(name)

has_value(name::String) = has_name(name)
has_node(name::String) = has_name(name)
has_graph(name::String) = has_name(name)

function dimension_name()
    ns = NAMESPACE[]

    i = 1
    while true
        name = try_create_dim(ns.dimensions, "", i)
        if !isnothing(name)
            return name
        end
        i += 1
    end
    return name
end

function with_prefix(f, prefix::String)
    ns = NAMESPACE[]
    new_ns = Namespace(get_name(prefix) * '/', ns.names, ns.dimensions) # TODO: Should not ns.prefix be here as well?
    return with(f, NAMESPACE => new_ns)
end

function get_name(name::String)
    ns = NAMESPACE[]

    base = ns.prefix * name
    c = 1
    fullname = base * string(c)
    while fullname ∈ ns.names
        c += 1
        fullname = base * string(c)
    end

    push!(ns.names, fullname)
    return fullname
end

function add_name(name::String)
    ns = NAMESPACE[]
    if name ∈ ns.names
        error("\"$name\" already exist in the namespace.")
    end
    push!(ns.names, name)

    return nothing
end

function has_name(name::String)
    ns = NAMESPACE[]
    return name ∈ ns.names
end

function try_create_dim(dimensions, prefix, depth)
    if depth == 0
        name = Symbol(prefix)
        if name ∉ dimensions
            push!(dimensions, name)
            return name
        end
        return nothing
    end
    for c in 'A':'Z'
        name = try_create_dim(dimensions, prefix * c, depth - 1)
        if !isnothing(name)
            return name
        end
    end
    return nothing
end

function _dim_name(i)
    alphabet = 'A':'Z'
    n = length(alphabet)
    name = ""
    while i > n
        ch = i % n
        name = alphabet[ch + 1] * name
        i = i ÷ n
    end
    if i > 0
        name = alphabet[i] * name
    end
    return Symbol(name)
end
