
function Mumps{Tv}(sym=0,par=1) where Tv
    mumps = Mumps{Tv}(sym,par,MPI.COMM_WORLD.val)
    invoke_mumps!(mumps)
    return mumps
end

function Mumps(A::AbstractArray{Tv},sym=issymmetric(A),par=1) where Tv<:Number
    mumps = Mumps{Tv}(sym,par)
    invoke_mumps!(mumps)
    provide_matrix!(mumps,A)
    return mumps
end

function LinearAlgebra.det(mumps::Mumps)
    if mumps.icntl[33]==0
        @warn "icntl[33]=0, determinant not computed"
        return complex(NaN,NaN)
    else
        return complex(mumps.rinfog[12],mumps.rinfog[13])*2^mumps.infog[34]
    end
end

# function get_matrix(mumps::Mumps{TR,TC}) where {TR,TC}
#     i = unsafe_wrap(Array{MUMPS_INT,1},mumps.irn,mumps.nnz; own=true)
#     j = unsafe_wrap(Array{MUMPS_INT,1},mumps.jcn,mumps.nnz; own=true)
#     a = unsafe_wrap(Array{TC,1},mumps.a,mumps.nnz; own=true)
#     return sparse(i,j,a,mumps.n,mumps.n)
# end
