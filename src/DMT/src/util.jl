global CHECK=false

import ITensors.scalar
export @cassert, @showinds, ⊗,scalar, arr1d, combine_by_prime,unzip
export embed_itensor_doubled
# conditional assert: if global flag CHECK is set, do the assert thing.
# Copy / pasted from julia 1.6.1.

macro cassert(ex, msgs...)
    if CHECK
        msg = isempty(msgs) ? ex : msgs[1]
        if isa(msg, AbstractString)
            msg = msg # pass-through
        elseif !isempty(msgs) && (isa(msg, Expr) || isa(msg, Symbol))
            # message is an expression needing evaluating
            msg = :(Main.Base.string($(esc(msg))))
        elseif isdefined(Main, :Base) && isdefined(Main.Base, :string) && applicable(Main.Base.string, msg)
            msg = Main.Base.string(msg)
        else
            # string() might not be defined during bootstrap
            msg = quote
                msg = $(Expr(:quote,msg))
                isdefined(Main, :Base) ? Main.Base.string(msg) :
                    (Core.println(msg); "Error during bootstrap. See stdout.")
            end
        end
        return :($(esc(ex)) ? $(nothing) : throw(AssertionError($msg)))
    end
end

macro showinds(nm)
    snm = string(nm)
    return :(println($snm); for j in inds($(esc(nm))) println(" $j") end)
end

scalar(v) = (@cassert length(v) == 1; v[1])

function trial_division_factorization(n :: Int)
    factors = []
    j = 2
    while j <= n
        while mod(n,j) == 0
            push!(factors, j)
            n /= j
        end
        j += 1
    end
    return factors |> Array{Int}
end

function integer_sqrt(prime_factors :: Array{Int})
    prime_factors = prime_factors |> sort
    
    a = b = 1
    
    while(length(prime_factors) >= 1)
        p = pop!(prime_factors)
        if a < b   a *= p
        else       b *= p
        end
    end
   
    return (a,b)
end

arr1d(x) = reshape(x, length(x))
⊗ = kron

#inds(H) .|> plev |> unique |> Set == [0,1] |> Set

function combine_by_prime(H :: ITensor)
    js = inds(H,plev=0)
    @cassert inds(H) |> Set == js ∪ js' |> Set
    Hc = deepcopy(H)
    C = combiner(js)
    Hc *= C
    Hc *= C'
    @cassert length(inds(Hc)) == 2
    return Hc
end

function combine_by_id(H :: ITensor)
    ids = id.(inds(B)) |> unique
    Hc = H
    for hex in ids
        jhex = filter(x -> id(x) == hex, J)
        C = combiner(jhex...)
        Hc = C*Hc
    end
    return Hc
end

unzip(A :: Array{Tuple{T,S}} where {T,S}) = ([a[1] for a in A], [a[2] for a in A])

function embed_itensor_doubled(A :: ITensor, ss :: Vector{<:Index})
    new_ss = setdiff(ss, inds(A))
    for s in new_ss
        A *= onehot(s => 1)
    end
    return A
end
 
