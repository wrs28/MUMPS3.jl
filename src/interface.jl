export set_icnlt!,
provide_matrix!,
provide_scaling!,
provide_perm_in!,
provide_rhs!


function set_icntl!(mumps::Mumps,i::Int,val::Int)
    icntl = mumps.icntl
    mumps.icntl = (icntl[1:i-1]...,val,icntl[i+1:end]...)
    return nothing
end


function provide_matrix!(mumps::Mumps{TR,TC},A::SparseMatrixCSC{TA}) where {TR,TC,TA}
    @assert A.m==A.n "matrix must be square, but it is $(A.n) by $(A.m)"
    if !(TC==TA)
        @warn "matrix with element type $TA will attempt be converted to Mumps type $TC"
    end
    if mumps.icntl[5]==1
        provide_matrix_elemental!(mumps,A)
    else
        provide_matrix_assembled!(mumps,A)
    end
    return nothing
end
function provide_matrix_assembled!(mumps::Mumps{TR,TC},A::SparseMatrixCSC{TA}) where {TR,TC,TA}
    if mumps.icntl[18] ∈ [1,2,3]
        provide_matrix_assembled_distributed!(mumps,A)
    else
        if mumps.sym ∈ [1,2]
            I, J, V = findnz(sparse(Symmetric(A)))
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
function provide_matrix_assembled_distributed!(mumps::Mumps{TR,TC},A::SparseMatrixCSC{TA}) where {TR,TC,TA}
    throw(ErrorException("not written yet."))
end
function provide_matrix_elemental!(mumps::Mumps{TR,TC},A::SparseMatrixCSC{TA}) where {TR,TC,TA}
    throw(ErrorException("not written yet."))
end


function provide_scaling!(mumps::Mumps{TR,TC},rowsca,colsca) where {TR,TC,TA}
    throw(ErrorException("not written yet."))
end
function provide_perm_in!(mumps::Mumps{TR,TC},rowsca,colsca) where {TR,TC,TA}
    throw(ErrorException("not written yet."))
end


function provide_rhs!(mumps::Mumps{TR,TC},rhs) where {TR,TC,TA}
    if mumps.icntl[20] ∈ [1,2,3]
        rhs = convert(SparseMatrixCSC,rhs)
        mumps.nz_rhs = length(rhs.nzval)
        mumps.nrhs = size(rhs,2)
        mumps.rhs_sparse = pointer(rhs.nzval)
        mumps.irhs_sparse = pointer(rhs.rowval)
        mumps.irhs_ptr = pointer(rhs.colptr)
        mumps.icntl[21]==0 ? mumps.rhs = pointer(Array{TC,2}(undef,size(rhs)...)) : nothing
    else
        rhs = convert(Matrix,rhs)
        mumps.rhs = pointer(convert(Matrix,rhs)[:])
        mumps.lrhs = size(rhs,1)
        mumps.nrhs = size(rhs,2)
    end
    return nothing
end
