"""
    @overload f forward first_type second_type N

Create methods for all possible calls to `f` with `N` arguments, where at least one
argument is of type `first_type`. The remaining arguments will have type `second_type`.
The methods will simply forward the call to the function `forward`.

# Examples
```julia
@overload Base.max _mymax ProbeArray AbstractArray 2
```
will produce the following methods
```julia
Base.max(A1::ProbeArray, A2::AbstractArray) = _mymax(A1, A2)
Base.max(A1::AbstractArray, A2::ProbeArray) = _mymax(A1, A2)
Base.max(A1::ProbeArray, A2::ProbeArray) = _mymax(A1, A2)
```
"""
macro overload(f, forward, first_type, second_type, N)
    N = eval(N)
    methods = Expr[]

    for mask in 1:(2 ^ N - 1)
        arglist = []
        callargs = []

        for i in 1:N
            if ((mask >> (i - 1)) & 1) == 1
                push!(arglist, :($(Symbol(:A, i))::$first_type))
            else
                push!(arglist, :($(Symbol(:A, i))::$second_type))
            end
            push!(callargs, Symbol(:A, i))
        end

        push!(
            methods, quote
                $f($(arglist...); kwargs...) = $forward($(callargs...); kwargs...)
            end
        )
    end

    return Expr(:block, methods...)
end

"""
    @overload_splat f forward first_type second_type N

Create methods for all possible calls to `f`, where at least one of the `N` first arguments
is of type `first_type`. The remaining arguments will have type `second_type`. The methods
will simply forward the call to the function `forward`.

# Examples
```julia
@overload_splat Base.max _mymax ProbeArray AbstractArray 2
```
will produce the following methods
```julia
Base.max(A1::ProbeArray, Arest::AbstractArray...) = _mymax(A1, Arest...)
Base.max(A1::AbstractArray, A2::ProbeArray, Arest::AbstractArray...) = _mymax(A1, A2, Arest...)
Base.max(A1::ProbeArray, A2::ProbeArray, Arest::AbstractArray...) = _mymax(A1, A2, Arest...)
```
"""
macro overload_splat(f, forward, first_type, second_type, N, first_arg=nothing)
    N = eval(N)
    methods = Expr[]

    for mask in 1:(2 ^ N - 1)
        n = 64 - leading_zeros(Int64(mask))

        arglist = []
        callargs = []

        if !isnothing(first_arg)
            push!(arglist, first_arg)
            push!(callargs, first_arg)
        end

        for i in 1:n
            if ((mask >> (i - 1)) & 1) == 1
                push!(arglist, :($(Symbol(:A, i))::$first_type))
            else
                push!(arglist, :($(Symbol(:A, i))::$second_type))
            end
            push!(callargs, Symbol(:A, i))
        end

        push!(arglist, :(Arest::$second_type...))
        push!(callargs, :(Arest...))

        push!(
            methods, quote
                $f($(arglist...); kwargs...) = $forward($(callargs...); kwargs...)
            end
        )
    end

    return Expr(:block, methods...)
end
