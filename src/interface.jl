# this file provides the low-level interface with the MUMPS 5.1.2 library
# by controlling access to the pointers contained in the Mumps object.
# Many functions are unsafe and are marked so.

export invoke_mumps_unsafe!,
set_icntl!,
provide_matrix!,
provide_rhs!,
get_rhs!,
get_sol!,
set_schur_centralized_by_column!,
get_schur!


"""
    invoke_mumps_unsafe!(mumps)

Call the appropriate mumps C-library, passing to it the Mumps object `mumps`

This is a low-level function, meaning that you have complete control over what operations are done,
based on the MUMPS manual.

Be warned, a direct call can crash Julia if `mumps` is not appropriately initialized.
"""
function invoke_mumps_unsafe!(mumps::Mumps{TC,TR}) where {TC,TR}
    @assert MPI.Initialized() "must call MPI.Init() exactly once before calling mumps"
    if TC==Float32
        cfun = :smumps_c
    elseif TC==Float64
        cfun = :dmumps_c
    elseif TC==ComplexF32
        cfun = :cmumps_c
    elseif TC==ComplexF64
        cfun = :zmumps_c
    end
    @eval ccall(($(string(cfun)),$(MUMPS_LIB)),
        Cvoid, (Ref{Mumps{$TC,$TR}},), $mumps)
    return nothing
end


"""
    set_icntl!(mumps,i,val; [displaylevel=1])

Set the integer control array according to ICNTL[i]=val
"""
function set_icntl!(mumps::Mumps,i::Int,val::Int; displaylevel=mumps.icntl[4]-1)
    icntl = mumps.icntl
    mumps.icntl = (icntl[1:i-1]...,convert(MUMPS_INT,val),icntl[i+1:end]...)
    displaylevel>0 ? display_icntl(mumps.icntl,i,val) : nothing
    return nothing
end


"""
    provide_matrix!(mumps,A)

Provide a square matrix `A` to a `mumps` object. It internally converts
`A` to be consistent with the ICNTL[5] setting.

If needed, it tries to convert element type of `A` to be consistent with
type of `mumps`, throwing a warning in this case.
"""
function provide_matrix!(mumps::Mumps{TC},A::AbstractArray{TA}) where {TC,TA}
    @assert size(A,1)==size(A,2) "matrix must be square, but it is $(size(A,1))×$(size(A,2))"
    if !(TC==TA)
        @warn "matrix with element type $TA will attempt be converted to Mumps type $TC"
    end
    if mumps.icntl[5]==1
        !(typeof(A)<:SparseMatrixCSC) ? nothing : (@warn "matrix is sparse, but ICNTL[5]=$(mumps.icntl[5]) indicates elemental. attempting to convert matrix to dense")
        _provide_matrix_elemental!(mumps,convert(Matrix,A))
    else
        typeof(A)<:SparseMatrixCSC ? nothing : (@warn "matrix is dense, but ICNTL[5]=$(mumps.icntl[5]) indicates assembled. attempting to convert matrix to sparse")
        _provide_matrix_assembled!(mumps,convert(SparseMatrixCSC,A))
    end
    return nothing
end
function _provide_matrix_assembled!(mumps::Mumps{TC},A::SparseMatrixCSC{TA}) where {TC,TA}
    if mumps.icntl[18] ∈ [1,2,3]
        _provide_matrix_assembled_distributed!(mumps,A)
    else
        if mumps.sym ∈ [1,2]
            I,J,V = findnz(sparse(Symmetric(A)))
        else
            I,J,V = findnz(A)
        end
        irn, jcn, a = convert.((Array{MUMPS_INT},Array{MUMPS_INT},Array{TC}),(I,J,V))
        mumps.irn, mumps.jcn, mumps.a = pointer.((irn,jcn,a))
        mumps.n = A.n
        mumps.nnz = length(A.nzval)
        return nothing
    end
end
function _provide_matrix_assembled_distributed!(mumps::Mumps{TC},A::SparseMatrixCSC{TA}) where {TC,TA}
    throw(ErrorException("not written yet."))
end
function _provide_matrix_elemental!(mumps::Mumps{TC},A::Array{TA}) where {TC,TA}
    mumps.n = size(A,1)
    mumps.nelt = 1
    mumps.eltptr = pointer(convert.(MUMPS_INT,[1,mumps.n+1]))
    mumps.eltvar = pointer(convert.(MUMPS_INT,collect(1:mumps.n)))
    if mumps.sym==0
        mumps.a_elt = pointer(convert.(TC,A[:]))
    else
        mumps.a_elt = pointer(convert.(TC,[A[i,j] for i ∈ 1:mumps.n for j ∈ 1:i]))
    end
    return nothing
end


"""
    provide_rhs!(mumps,rhs)

Provide a RHS matrix or vector `rhs` to a `mumps` object. It internally converts
`rhs` to be consistent with the ICNTL[20] setting, and additionally allocates
`mumps.rhs` according to the ICNTL[21] setting.

If needed, it tries to convert element type of `rhs` to be consistent with
type of `mumps`, throwing a warning in this case.
"""
function provide_rhs!(mumps::Mumps{TC},rhs::AbstractMatrix{TA}) where {TC,TA}
    if mumps.icntl[20] ∈ [1,2,3]
        rhs = convert(SparseMatrixCSC,rhs)
        mumps.nz_rhs = length(rhs.nzval)
        mumps.nrhs = size(rhs,2)
        mumps.rhs_sparse = pointer(convert.(TC,rhs.nzval))
        mumps.irhs_sparse = pointer(convert.(MUMPS_INT,rhs.rowval))
        mumps.irhs_ptr = pointer(convert.(MUMPS_INT,rhs.colptr))
        if mumps.icntl[21]==0
            mumps.rhs = pointer(fill(convert(TC,NaN),prod(size(rhs))))
            mumps.lrhs = size(rhs,1)
        end
    else
        rhs = convert(Matrix,rhs)
        mumps.rhs = pointer(convert(Matrix,rhs)[:])
        mumps.lrhs = size(rhs,1)
        mumps.nrhs = size(rhs,2)
    end
    return nothing
end
function provide_rhs!(mumps,rhs::AbstractVector)
    provide_rhs!(mumps,repeat(rhs,1,1))
    return nothing
end


"""
    get_rhs!(x,mumps)

Retrieve right hand side from `mumps`, storing it in pre-allocated `x`
"""
function get_rhs!(x,mumps::Mumps)
    initialized(mumps)
    if has_rhs(mumps)
        get_rhs_unsafe!(x,mumps)
    else
        @warn "mumps has no rhs"
    end
    return nothing
end
function get_rhs_unsafe!(x::SparseMatrixCSC,mumps::Mumps)
    @assert mumps.icntl[20] ∈ [1,2,3] "rhs is dense, target is sparse. try with dense target"
    for i ∈ LinearIndices(x.colptr)
        x.colptr[i] = unsafe_load(mumps.irhs_ptr,i)
    end
    for i ∈ LinearIndices(x.rowval)
        x.rowval[i] = unsafe_load(mumps.irhs_sparse,i)
        x.nzval[i] = unsafe_load(mumps.rhs_sparse,i)
    end
    return nothing
end
function get_rhs_unsafe!(x::Array,mumps::Mumps)
    @assert !(mumps.icntl ∈ [1,2,3]) "rhs is sparse, target is dense. try with sparse target"
    for i ∈ LinearIndices(x)
        x[i] = unsafe_load(mumps.rhs,i)
    end
    return nothing
end

get_sol!(x::Array,mumps::Mumps) = get_rhs!(x,mumps)


"""
    get_schur!(S,mumps)

Retrieve Schur complement matrix from `mumps` into pre-allocated `S`
"""
function get_schur!(S,mumps::Mumps)
    @assert mumps.size_schur>0 "schur complement not yet allocated."
    for i ∈ LinearIndices(S)
        S[i] = unsafe_load(mumps.schur,i)
    end
    return nothing
end


function LinearAlgebra.det(mumps::Mumps{TC}) where TC
    if mumps.icntl[33]==0
        @warn "icntl[33]=0, determinant not computed"
        d = TC<:Complex ? complex(NaN,NaN) : NaN
    else
        d = TC<:Complex ? complex(mumps.rinfog[12],mumps.rinfog[13])*2^mumps.infog[34] : mumps.rinfog[12]*2^mumps.infog[34]
    end
    return convert(TC,d)
end




# function provide_scaling!(mumps::Mumps{TC},rowsca,colsca) where {TC,TA}
#     throw(ErrorException("not written yet."))
# end
# function provide_perm_in!(mumps::Mumps{TC},rowsca,colsca) where {TC,TA}
#     throw(ErrorException("not written yet."))
# end
