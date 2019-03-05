# this file provides the low-level interface with the MUMPS 5.1.2 library
# by controlling access to the pointers contained in the Mumps object.
# Many functions are unsafe and are marked as such.
# None of these functions change the JOB parameter.

export invoke_mumps!,
set_icntl!, set_job!,
provide_matrix!,
provide_rhs!,
get_rhs!, get_rhs,
get_sol!, get_sol,
set_schur_centralized_by_column!,
get_schur!, get_schur


"""
    invoke_mumps_unsafe!(mumps)

Call the appropriate mumps C-library, passing to it the Mumps object `mumps`

This is a low-level function, meaning that you have complete control over what
operations are done, based on the MUMPS manual.

Be warned, a direct call can crash Julia if `mumps` is not appropriately
initialized.

See also: [`invoke_mumps!`](@ref)
"""
invoke_mumps_unsafe!(mumps::Mumps) = invoke_mumps_unsafe!(mumps.mumpsc)
@inline function invoke_mumps_unsafe!(mumpsc::MumpsC{TC,TR}) where {TC,TR}
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
    sym = dlsym(LIB,cfun)
    ccall(sym,Cvoid,(Ref{MumpsC{TC,TR}},), mumpsc)
    return nothing
end
"""
    invoke_mumps!(mumps)

Call the appropriate mumps C-library, passing to it the Mumps object `mumps`,
but checking to make sure `mumps` has been initialized first, so that it's safe.

This is a low-level function, meaning that you have complete control over what
operations are done, based on the MUMPS manual.

Be warned, a direct call can crash Julia if `mumps` is not appropriately
initialized.

See also: [`invoke_mumps_unsafe!`](@ref)
"""
function invoke_mumps!(mumps::Mumps)
    check_finalized(mumps)
    invoke_mumps_unsafe!(mumps)
end
check_finalized(mumps::Mumps) = @assert !mumps.finalized "Mumps object already finalized"


"""
    set_icntl!(mumps,i,val; [displaylevel=1])

Set the integer control array according to ICNTL[i]=val

See also: [`display_icntl`](@ref)
"""
function set_icntl!(mumps::Mumps,i::Int,val::Int; displaylevel=mumps.mumpsc.icntl[4]-1)
    icntl = mumps.mumpsc.icntl
    mumps.mumpsc.icntl = (icntl[1:i-1]...,convert(MUMPS_INT,val),icntl[i+1:end]...)
    displaylevel>0 ? display_icntl(mumps.mumpsc.icntl,i,val) : nothing
    return nothing
end


set_job!(mumps::Mumps,i) = mumps.mumpsc.job=i


"""
    provide_matrix!(mumps,A)

Provide a square matrix `A` to a `mumps` object. It internally converts
`A` to be consistent with the ICNTL[5] setting.

If needed, it tries to convert element type of `A` to be consistent with
type of `mumps`, throwing a warning in this case.

See also: [`provide_rhs!`](@ref)
"""
function provide_matrix!(mumps::Mumps{T},A::AbstractArray{TA}) where {T,TA}
    @assert size(A,1)==size(A,2) "matrix must be square, but it is $(size(A,1))×$(size(A,2))"
    if !(T==TA)
        @warn "matrix with element type $TA will attempt be converted to Mumps type $T"
    end
    if is_matrix_assembled(mumps)
        typeof(A)<:SparseMatrixCSC ? nothing : (@warn "matrix is dense, but ICNTL[5]=$(mumps.mumpsc.icntl[5]) indicates assembled. attempting to convert matrix to sparse")
        _provide_matrix_assembled!(mumps,convert(SparseMatrixCSC,A))
    else
        !(typeof(A)<:SparseMatrixCSC) ? nothing : (@warn "matrix is sparse, but ICNTL[5]=$(mumps.mumpsc.icntl[5]) indicates elemental. attempting to convert matrix to dense")
        _provide_matrix_elemental!(mumps,convert(Matrix,A))
    end
end
function _provide_matrix_assembled!(mumps::Mumps,A::SparseMatrixCSC{T}) where T
    if is_matrix_distributed(mumps)
        _provide_matrix_assembled_distributed!(mumps,A)
    else
        _provide_matrix_assembled_centralized!(mumps,A)
    end
end
function _provide_matrix_assembled_centralized!(mumps::Mumps{T},A::SparseMatrixCSC) where T
    mumpsc, gc_haven = mumps.mumpsc, mumps.gc_haven
    if is_symmetric(mumps)
        I,J,V = findnz(triu(A))
    else
        I,J,V = findnz(A)
    end
    irn, jcn, a = convert.((Array{MUMPS_INT},Array{MUMPS_INT},Array{T}),(I,J,V))
    gc_haven.irn, gc_haven.jcn, gc_haven.a = irn, jcn, a
    mumpsc.irn, mumpsc.jcn, mumpsc.a = pointer.((irn,jcn,a))
    mumpsc.n = A.n
    mumpsc.nnz = length(V)
    return nothing
end
function _provide_matrix_assembled_distributed!(mumps::Mumps{T},A::SparseMatrixCSC) where T
    throw(ErrorException("not written yet."))
    return nothing
end
function _provide_matrix_elemental!(mumps::Mumps{T},A::Array) where T
    mumpsc, gc_haven = mumps.mumpsc, mumps.gc_haven
    mumpsc.n = size(A,1)
    mumpsc.nelt = 1
    eltptr = convert.(MUMPS_INT,[1,mumpsc.n+1])
    eltvar = convert.(MUMPS_INT,collect(1:mumpsc.n))
    gc_haven.eltptr, gc_haven.eltvar = eltptr, eltvar
    mumpsc.eltptr, mumpsc.eltvar = pointer(eltptr), pointer(eltvar)
    if is_symmetric(mumps)
        a_elt = convert.(T,[A[i,j] for i ∈ 1:mumpsc.n for j ∈ 1:i])
    else
        a_elt = convert.(T,A[:])
    end
    gc_haven.a_elt = a_elt
    mumpsc.a_elt = pointer(a_elt)
    return nothing
end


"""
    provide_rhs!(mumps,y)

Provide a RHS matrix or vector `rhs` to a `mumps` object. It internally converts
`rhs` to be consistent with the ICNTL[20] setting, and additionally allocates
`mumps.rhs` according to the ICNTL[21] setting.

If needed, it tries to convert element type of `rhs` to be consistent with
type of `mumps`, throwing a warning in this case.

See also: [`provide_matrix!`](@ref)
"""
function provide_rhs!(mumps::Mumps,rhs::AbstractMatrix)
    if is_rhs_dense(mumps)
        provide_rhs_dense!(mumps,rhs)
    else
        provide_rhs_sparse!(mumps,rhs)
    end
end
function provide_rhs_sparse!(mumps::Mumps{T},rhs::AbstractMatrix) where T
    gc_haven, mumpsc = mumps.gc_haven, mumps.mumpsc
    rhs = convert(SparseMatrixCSC,rhs)

    mumpsc.nz_rhs = length(rhs.nzval)
    mumpsc.nrhs = size(rhs,2)

    rhs_sparse = convert.(T,rhs.nzval)
    irhs_sparse = convert.(MUMPS_INT,rhs.rowval)
    irhs_ptr = convert.(MUMPS_INT,rhs.colptr)

    gc_haven.rhs_sparse = rhs_sparse
    gc_haven.irhs_sparse = irhs_sparse
    gc_haven.irhs_ptr = irhs_ptr

    mumpsc.rhs_sparse = pointer(rhs_sparse)
    mumpsc.irhs_sparse = pointer(irhs_sparse)
    mumpsc.irhs_ptr = pointer(irhs_ptr)
    if is_sol_central(mumps)
        y = fill(convert(T,NaN),prod(size(rhs)))
        gc_haven.rhs = y
        mumpsc.rhs = pointer(y)
        mumpsc.lrhs = size(rhs,1)
    end
    return nothing
end
function provide_rhs_dense!(mumps::Mumps{T},rhs::AbstractMatrix) where T
    gc_haven, mumpsc = mumps.gc_haven, mumps.mumpsc
    y = convert(Matrix{T},rhs)[:]
    gc_haven.rhs = y
    mumpsc.rhs = pointer(y)
    mumpsc.lrhs = size(rhs,1)
    mumpsc.nrhs = size(rhs,2)
    return nothing
end
provide_rhs!(mumps::Mumps,rhs::AbstractVector) = provide_rhs!(mumps,repeat(rhs,1,1))


"""
    get_rhs!(y,mumps)

Retrieve right hand side from `mumps`, storing it in pre-allocated `x`

See also: [`get_rhs`](@ref), [`get_sol!`](@ref), [`get_sol`](@ref)
"""
function get_rhs!(x,mumps::Mumps)
    check_finalized(mumps)
    if has_rhs(mumps)
        get_rhs_unsafe!(x,mumps)
    else
        @warn "mumps has no rhs"
    end
    return nothing
end
function get_rhs_unsafe!(x::SparseMatrixCSC,mumps::Mumps)
    @assert !is_rhs_dense(mumps) "rhs is dense, target is sparse. try with dense target"
    for i ∈ LinearIndices(x.colptr)
        x.colptr[i] = unsafe_load(mumps.mumpsc.irhs_ptr,i)
    end
    for i ∈ LinearIndices(x.rowval)
        x.rowval[i] = unsafe_load(mumps.mumpsc.irhs_sparse,i)
        x.nzval[i] = unsafe_load(mumps.mumpsc.rhs_sparse,i)
    end
    return nothing
end
function get_rhs_unsafe!(x::Array,mumps::Mumps)
    @assert is_rhs_dense(mumps) "rhs is sparse, target is dense. try with sparse target"
    for i ∈ LinearIndices(x)
        x[i] = unsafe_load(mumps.mumpsc.rhs,i)
    end
    return nothing
end
"""
    get_rhs(mumps) -> y

Retrieve right hand side from `mumps`

See also: [`get_rhs!`](@ref), [`get_sol!`](@ref), [`get_sol`](@ref)
"""
function get_rhs(mumps::Mumps{T}) where T
    n = mumps.mumpsc.nrhs
    if !is_rhs_dense(mumps)
        m = mumps.mumpsc.n
        colptr = ones(MUMPS_INT,mumps.mumpsc.nrhs+1)
        rowval = ones(MUMPS_INT,mumps.mumpsc.nz_rhs)
        nzval = Array{T}(undef,mumps.mumpsc.nz_rhs)
        x = SparseMatrixCSC(m,n,colptr,rowval,nzval)
    else
        m = mumps.mumpsc.lrhs
        x = Array{T}(undef,m,n)
    end
    get_rhs!(x,mumps)
    return x
end


"""
    get_sol!(x,mumps)

Retrieve solution `x` from `mumps` into pre-allocated array.

See also: [`get_rhs!`](@ref), [`get_rhs`](@ref), [`get_sol`](@ref)
"""
function get_sol!(x::Array,mumps::Mumps)
    check_finalized(mumps)
    if mumps.mumpsc.job ∉ [3,5,6]
        @warn "mumps has not passed through a solution phase"
    end
    if has_rhs(mumps)
        get_sol_unsafe!(x,mumps)
    else
        @warn "mumps has no rhs"
    end
    return nothing
end
function get_sol_unsafe!(x::Array,mumps::Mumps)
    for i ∈ LinearIndices(x)
        x[i] = unsafe_load(mumps.mumpsc.rhs,i)
    end
    return nothing
end
"""
    get_sol(mumps) -> x

Retrieve solution from `mumps`

See also: [`get_rhs!`](@ref), [`get_rhs`](@ref), [`get_sol!`](@ref)
"""
function get_sol(mumps::Mumps{T}) where T
    if mumps.mumpsc.job ∉ [3,5,6]
        @warn "mumps has not passed through a solution phase"
    end
    x = Array{T}(undef,mumps.mumpsc.lrhs,mumps.mumpsc.nrhs)
    get_sol!(x,mumps)
    return x
end


"""
    get_schur!(S,mumps)

Retrieve Schur complement matrix from `mumps` into pre-allocated `S`

See also: [`get_schur`](@ref), [`mumps_schur!`](@ref), [`mumps_schur`](@ref)
"""
function get_schur!(S,mumps::Mumps)
    @assert has_schur(mumps) "schur complement not yet allocated."
    get_schur_unsafe!(S,mumps)
end
function get_schur_unsafe!(S,mumps::Mumps)
    for i ∈ LinearIndices(S)
        S[i] = unsafe_load(mumps.mumpsc.schur,i)
    end
    return nothing
end
"""
    get_schur(mumps) -> S

Retrieve Schur complement matrix `S` from `mumps`

See also: [`get_schur!`](@ref), [`mumps_schur!`](@ref), [`mumps_schur`](@ref)
"""
function get_schur(mumps::Mumps{T}) where T
    S = Array{T}(undef,mumps.mumpsc.size_schur,mumps.mumpsc.size_schur)
    get_schur!(S,mumps)
    return S
end


"""
    set_schur_centralized_by_column!(mumps,schur_inds)

Set up Schur complement matrix calculation for the "centralized by column"
method suggested in the MUMS manual

See also: [`mumps_schur!`](@ref), [`mumps_schur`](@ref)
"""
function set_schur_centralized_by_column!(mumps::Mumps{T},schur_inds::AbstractArray{Int}) where T
    gc_haven, mumpsc = mumps.gc_haven, mumps.mumpsc
    mumpsc.size_schur = length(schur_inds)
    listvar_schur = convert.(MUMPS_INT,schur_inds)
    gc_haven.listvar_schur = listvar_schur
    mumpsc.listvar_schur = pointer(listvar_schur)
    mumpsc.nprow = 1
    mumpsc.npcol = 1
    mumpsc.mblock = 100
    mumpsc.nblock = 100
    mumpsc.schur_lld = mumpsc.size_schur
    schur = Array{T}(undef,mumps.mumpsc.size_schur^2)
    gc_haven.schur = schur
    mumpsc.schur = pointer(schur)
    set_icntl!(mumps,19,3)
end


function LinearAlgebra.det(mumps::Mumps{T}) where T
    if !has_det(mumps)
        @warn "ICNTL[33]=0, determinant not computed"
        d = T<:Complex ? complex(NaN,NaN) : NaN
    else
        if T<:Complex
            d = complex(mumps.mumpsc.rinfog[12],mumps.mumpsc.rinfog[13])*2^mumps.mumpsc.infog[34]
        else
            d = mumps.mumpsc.rinfog[12]*2^mumps.mumpsc.infog[34]
        end
    end
    return convert(T,d)
end


####################################################################
### auxilliary functions
####################################################################
is_matrix_assembled(mumps::Mumps) = is_matrix_assembled(mumps.mumpsc)
is_matrix_assembled(mumpsc::MumpsC) = !(mumpsc.icntl[5] ∈ [1])

is_matrix_distributed(mumps::Mumps) = is_matrix_distributed(mumps.mumpsc)
is_matrix_distributed(mumpsc::MumpsC) = mumpsc.icntl[18] ∈ [1,2,3]

is_rhs_dense(mumps::Mumps) = is_rhs_dense(mumps.mumpsc)
is_rhs_dense(mumpsc::MumpsC) = mumpsc.icntl[20] ∉ [1,2,3]

is_sol_central(mumps::Mumps) = is_sol_central(mumps.mumpsc)
is_sol_central(mumpsc::MumpsC) = mumpsc.icntl[21] ∉ [1]

has_det(mumps::Mumps) = has_det(mumps.mumpsc)
has_det(mumpsc::MumpsC) = mumpsc.icntl[33] ∉ [0]

is_symmetric(mumps::Mumps) = is_symmetric(mumps.mumpsc)
is_symmetric(mumpsc::MumpsC) = mumpsc.sym ∈ [1,2]

is_posdef(mumps::Mumps) = is_posdef(mumps.mumpsc)
is_posdef(mumpsc::MumpsC) = mumpsc.sym ∈ [1]

has_matrix(mumps::Mumps) = has_matrix(mumps.mumpsc)
has_matrix(mumpsc::MumpsC) = mumpsc.n > 0

has_rhs(mumps::Mumps) = has_rhs(mumps.mumpsc)
has_rhs(mumpsc::MumpsC) = mumpsc.nrhs*mumpsc.lrhs>0 || mumpsc.nz_rhs>0

has_schur(mumps::Mumps) = has_schur(mumps.mumpsc)
has_schur(mumpsc::MumpsC) = mumpsc.size_schur > 0

LinearAlgebra.issymmetric(mumps::Mumps) = is_symmetric(mumps)
LinearAlgebra.issymmetric(mumpsc::MumpsC) = is_symmetric(mumpsc)
LinearAlgebra.isposdef(mumps::Mumps) = is_posdef(mumps)
LinearAlgebra.isposdef(mumpsc::MumpsC) = is_posdef(mumpsc)
