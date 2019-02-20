# this file contains useful MUMPS interaction tools which are
# not as low-level (and therefore generally safer) than those
# contained in "interface.jl"

export invoke_mumps!,
finalize!,
get_rhs,
get_sol,
get_schur


"""
    invoke_mumps!(mumps)

Call the appropriate mumps C-library, passing to it the Mumps object `mumps`,
but checking to make sure `mumps` has been initialized first, so that it's safe.

This is a low-level function, meaning that you have complete control over what operations are done,
based on the MUMPS manual.

Be warned, a direct call can crash Julia if `mumps` is not appropriately initialized.
"""
function invoke_mumps!(mumps::Mumps)
    initialized(mumps)
    invoke_mumps_unsafe!(mumps)
    return nothing
end


"""
    finalize!(mumps)

Release the pointers contained in `mumps`
"""
function finalize!(mumps::Mumps)
    if initialized(mumps)
        finalize_unsafe!(mumps)
    else
        @warn "mumps not initialized, doing nothing"
    end
    return nothing
end
function finalize_unsafe!(mumps::Mumps)
    mumps.job=-2
    invoke_mumps_unsafe!(mumps)
    return nothing
end


"""
    initialized(mumps)

Check whether `mumps` has been initialized, needed for ensuring safe operation elsewhere
"""
function initialized(mumps::Mumps)
    @assert mumps.job>-2 "accessing a Mumps object after being finalized will crash julia"
    return nothing
end
"""
    has_matrix(mumps) -> Bool

check whether `mumps` has been provided a matrix
"""
has_matrix(mumps::Mumps) = mumps.n>0
"""
    has_rhs(mumps) -> Bool

check whether `mumps` has been provided a right hand side
"""
has_rhs(mumps::Mumps) = mumps.nrhs*mumps.lrhs>0 || mumps.nz_rhs>0


"""
    get_rhs(mumps) -> rhs

Retrieve right hand side from `mumps`
"""
function get_rhs(mumps::Mumps{TC}) where TC
    n = mumps.nrhs
    if mumps.icntl[20] ∈ [1,2,3]
        m = mumps.n
        colptr = ones(MUMPS_INT,mumps.nrhs+1)
        rowval = ones(MUMPS_INT,mumps.nz_rhs)
        nzval = Array{TC}(undef,mumps.nz_rhs)
        x = SparseMatrixCSC(m,n,colptr,rowval,nzval)
    else
        m = mumps.lrhs
        x = Array{TC}(undef,m,n)
    end
    get_rhs!(x,mumps)
    return x
end


"""
    get_sol(mumps) -> x

Retrieve solution from `mumps`
"""
function get_sol(mumps::Mumps{TC}) where TC
    if mumps.job ∉ [3,5,6]
        @warn "mumps has not passed through a solution phase"
    end
    x = Array{TC}(undef,mumps.lrhs,mumps.nrhs)
    get_sol!(x,mumps)
    return x
end


"""
    get_schur(mumps) -> S

Retrieve Schur complement matrix `S` from `mumps`
"""
function get_schur(mumps::Mumps{TC}) where TC
    S = Array{TC}(undef,mumps.size_schur,mumps.size_schur)
    get_schur!(S,mumps)
    return S
end


"""
    set_schur_centralized_by_column!(mumps,schur_inds)

Set up Schur complement matrix calculation for the "centralized by column"
method suggested in the MUMS manual
"""
function set_schur_centralized_by_column!(mumps::Mumps{TC},schur_inds::Array{Int}) where TC
    mumps.size_schur = length(schur_inds)
    mumps.listvar_schur = pointer(convert.(MUMPS_INT,schur_inds))
    mumps.nprow = 1
    mumps.npcol = 1
    mumps.mblock = 100
    mumps.nblock = 100
    mumps.schur_lld = mumps.size_schur
    mumps.schur = pointer(Array{TC}(undef,mumps.size_schur^2))
    set_icntl!(mumps,19,3)
    return nothing
end
