# this file contains the main high-level functionality of MUMPS3:
# functions to solve A*x=y, LU factorization of A, compute the determinant
# of A, get a Schur complement matrix associated with A, compute the inverse
# of said Schur complemet matrix, compute select elements of the
# inverse of A

export mumps_solve!, mumps_solve,
mumps_factorize!, mumps_factorize,
mumps_det!, mumps_det,
mumps_schur!, mumps_schur,
mumps_select_inv!, mumps_select_inv,
mumps_inv


"""
    Mumps(A; sym=0, par=1) -> mumps
    Mumps(A, rhs; sym=0, par=1) -> mumps

Create an instance of a `Mumps` object with the same type as matrix `A`
and simultaneously provide `A` to it.

Optional arguments `sym` and `par` correspond to the MUMPS 5.1.2 paremeters.
They are:
    `sym` = 0 for unsymmetric, 1 for symm pos def, 2 for general symm
    `par` = 0 for host participating in parallel calculation, 1 otherwise
"""
function Mumps(A::AbstractArray{Tv}; kwargs...) where Tv<:Number
    if !haskey(kwargs,:sym)
        if issymmetric(A) && isposdef(A)
            kwargs = (kwargs...,:sym => 1)
        elseif issymmetric(A)
            kwargs = (kwargs...,:sym => 2)
        else
            kwargs = (kwargs...,:sym => 0)
        end
    end
    mumps = Mumps{Tv}(;kwargs...)
    invoke_mumps!(mumps)
    provide_matrix!(mumps,A)
    return mumps
end
function Mumps(A::AbstractArray{Tv}, rhs::AbstractArray{Tu}; sym=issymmetric(A), kwargs...) where {Tv,Tu}
    if !haskey(kwargs,:sym)
        if issymmetric(A) && isposdef(A)
            kwargs = (kwargs...,:sym => 1)
        elseif issymmetric(A)
            kwargs = (kwargs...,:sym => 2)
        else
            kwargs = (kwargs...,:sym => 0)
        end
    end
    mumps = Mumps{promote_type(Tv,Tu)}(;kwargs...)
    invoke_mumps!(mumps)
    provide_matrix!(mumps,A)
    if typeof(rhs)<:SparseMatrixCSC
        set_icntl!(mumps,20,1)
    elseif typeof(rhs)<:Array
        set_icntl!(mumps,20,0)
    else
        throw(ArgumentError("unrecognized array type for rhs"))
    end
    provide_rhs!(mumps,rhs)
    return mumps
end
"""
    Mumps{T}(; sym=0, par=1) -> mumps

Create an instance of `Mumps`, intialized.
"""
function Mumps{Tv}(;sym=0,par=1) where Tv
    mumps = Mumps{Tv}(sym,par,MPI.COMM_WORLD.val)
    invoke_mumps!(mumps)
    return mumps
end


"""
    mumps_solve!(x,mumps)
    mumps_solve!(x,A,y)
    mumps_solve!(x,mumps,y)

Solve `A*x=y`, saving result in pre-allocated x.
`mumps` must have previously been provided a matrix `A`.
If `y` is not given, `mumps` must have previously been provided `y`
"""
function mumps_solve!(x::Array,mumps::Mumps)
    initialized(mumps)
    @assert has_matrix(mumps) "matrix not yet provided to mumps object"
    @assert has_rhs(mumps) "rhs not yet provided to mumps object"
    if mumps.job ∈ [2,4] # if already factored, just solve
        mumps.job = 3
    elseif mumps.job==1 # if analyzed only, factorize and solve
        mumps.job=5
    elseif mumps.job ∈ [3,5] # is solved already, retrieve solution
        get_sol!(x,mumps)
        return nothing
    else # else analyze, factor, solve
        mumps.job=6
    end
    invoke_mumps!(mumps)
    get_sol!(x,mumps)
    return nothing
end
function mumps_solve!(x::Array,A::AbstractArray,rhs::AbstractArray)
    mumps = Mumps(A,rhs)
    suppress_display!(mumps)
    mumps_solve!(x,mumps)
    finalize!(mumps)
    return nothing
end
function mumps_solve!(x::Array,mumps::Mumps,rhs::AbstractArray)
    provide_rhs!(mumps,rhs)
    mumps.job=2
    mumps_solve!(x,mumps)
    return nothing
end
"""
    mumps_solve(mumps) -> x
    mumps_solve(A,y) -> x
    mumps_solve(mumps,y) -> x

Solve `A*x=y`
`mumps` must have previously been provided a matrix `A`.
If only input is `mumps` must also have been provided `y`.
"""
function mumps_solve(mumps::Mumps{TC}) where TC
    if mumps.job ∈ [3,5,6]
        x = get_sol(mumps)
    else
        x = get_rhs(mumps)
    end
    x = convert(Matrix,x)
    mumps_solve!(x,mumps)
    return x
end
function mumps_solve(A::AbstractArray,rhs::AbstractArray)
    x = convert(Matrix,rhs)
    mumps_solve!(x,A,rhs)
    return x
end
function mumps_solve(mumps::Mumps,rhs::AbstractArray)
    x = convert(Matrix,rhs)
    mumps_solve!(x,mumps,rhs)
    return x
end


"""
    mumps_factorize!(mumps)
    mumps_factorize(A) -> mumps

LU factorize `A`. LU stored in `mumps`, but not in a particularly accessible way.
Useful for wrapping in a linear map.
"""
function mumps_factorize!(mumps::Mumps)
    initialized(mumps)
    @assert has_matrix(mumps) "matrix not yet provided to mumps object"
    if mumps.job ∈ [2,3,4,5,6] # already factored
        @warn "already factored"
        return nothing
    elseif mumps.job==1 # if analyzed only, factorize
        mumps.job=2
    else # else analyze, factor
        mumps.job=4
    end
    invoke_mumps!(mumps)
    return nothing
end
function mumps_factorize(A::AbstractArray)
    mumps = Mumps(A)
    suppress_display!(mumps)
    mumps_factorize!(mumps)
    return mumps
end


"""
    mumps_det!(mumps; discard=true)
    mumps_det(A) -> det

Compute determinant of `A`, which has been previously provided to
`mumps`.

For `mumps_det!`, determinant can be computed after by just `det(mumps)`
[must have loaded LinearAlgebra].

Optional keyward `discard` controls whether LU factors are discarded via
ICNTL[31]. This is useful if you only care about determinant and don't
want to do any further computation with mumps. Use `discard=2` to throw
away only L. See manual for MUMPS 5.1.2.
"""
function mumps_det!(mumps::Mumps; discard=true)
    initialized(mumps)
    @assert has_matrix(mumps) "matrix not yet provided to mumps object"
    set_icntl!(mumps,31,Int(discard))
    set_icntl!(mumps,33,1)
    mumps.job > 0 ? mumps.job=1 : nothing
    mumps_factorize!(mumps)
    return nothing
end
function mumps_det(A)
    mumps = Mumps(A)
    suppress_display!(mumps)
    mumps_det!(mumps)
    finalize(mumps)
    return det(mumps)
end


"""
    mumps_schur!(mumps, schur_inds)
    mumps_schur!(mumps, x)

where `x` is sparse, schur indices determined from populated rows of `x`
"""
function mumps_schur!(mumps::Mumps, schur_inds::AbstractArray{Int,1})
    initialized(mumps)
    @assert has_matrix(mumps) "matrix not yet provided to mumps object"
    set_schur_centralized_by_column!(mumps, schur_inds)
    set_icntl!(mumps,7,0) # for now can't leave up to system because it could choose METIS (option 5) which has issues with the current build of mumps from brew (as of Feb 10 2019)
    if mumps.job==1 # if analyzed only, factorize
        mumps.job=2
    else # else analyze, factor
        mumps.job=4
    end
    invoke_mumps!(mumps)
    return nothing
end
mumps_schur!(mumps::Mumps, x::SparseMatrixCSC) = mumps_schur!(mumps,unique!(sort!(x.rowval)))
mumps_schur!(mumps::Mumps, x::SparseVector) = mumps_schur!(mumps,x.nzind)
"""
    mumps_schur(A,schur_inds) -> S
    mumps_schur(A,x) -> S

where `x` is sparse.
`S` is Schur complement matrix.
"""
function mumps_schur(A::AbstractArray,x)
    mumps = Mumps(A)
    suppress_display!(mumps)
    mumps_schur!(mumps,x)
    S = get_schur(mumps)
    finalize!(mumps)
    return S
end


"""
    mumps_select_inv!(x,mumps)
    mumps_select_inv!(x,A)

Compute selected elements of A⁻¹ with same sparsity pattern as `x`,
stored in `x`. If passed `mumps`, must have previously been provided
with matrix `A`.
"""
function mumps_select_inv!(x::AbstractSparseArray,mumps::Mumps)
    set_icntl!(mumps,30,1)
    set_icntl!(mumps,20,3)
    provide_rhs!(mumps,x)
    mumps_solve!(x,mumps)
    return nothing
end
function mumps_select_inv!(x,A)
    @assert size(x)==size(A) "target and matrix must have same size"
    mumps = Mumps(A)
    mumps_select_inv!(x,mumps)
    finalize!(mumps)
    return nothing
end
"""
    mumps_select_inv(A,I,J)

Compute selected elements of A⁻¹. The integer arrays
`I` and `J` specify which entries via `i(k),j(k) = I[k],J[k]`.
"""
function mumps_select_inv(A,I::AbstractArray{Int},J::AbstractArray{Int})
    x = sparse(I,J,fill(one(eltype(A)),length(I)),size(A)...)
    mumps_select_inv!(x,A)
    return x
end


"""
    mumps_inv(A) -> A⁻¹, mumps

compute LU factorization of A and return a Linear Map
equavalent to inv(A), as defined by their action on a
vector. Notably, A⁻¹ is not a matrix, and its elements
cannot be accessed directly.
"""
function mumps_inv(A)
    mumps = Mumps(A)
    if mumps.sym==1
        sym=true
        posdef=true
    elseif mumps.sym==2
        sym=true
        posdef=false
    else
        sym=false
        posdef=false
    end
    A⁻¹ = LinearMap{eltype(A)}((y,x)->mumps_solve!(y,mumps,x),convert(Int,mumps.n), ismutating=true, issymmetric=sym, isposdef=posdef)
    return A⁻¹, mumps
end



function Base.:\(mumps::Mumps,y)
    suppress_display!(mumps)
    if mumps.job < 2
        mumps_factorize!(mumps)
    end
    x = mumps_solve(mumps,y)
    return x
end
function LinearAlgebra.ldiv!(mumps::Mumps,y)
    suppress_display!(mumps)
    if mumps.job < 2
        mumps_factorize!(mumps)
    end
    provide_rhs!(mumps,y)
    mumps_solve!(y,mumps)
    return nothing
end
function LinearAlgebra.ldiv!(x,mumps::Mumps,y)
    suppress_display!(mumps)
    if mumps.job < 2
        mumps_factorize!(mumps)
    end
    provide_rhs!(mumps,y)
    mumps_solve!(x,mumps)
    return nothing
end
function LinearAlgebra.inv(mumps::Mumps)
    y = sparse(1:mumps.n,1:mumps.n,1:mumps.n,mumps.n,mumps.n)
    mumps_select_inv!(y,mumps)
    finalize(mumps)
    return y
end
