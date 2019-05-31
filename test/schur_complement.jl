using MUMPS3
using MPI, LinearAlgebra, SparseArrays

MPI.Initialized() ? nothing : MPI.Init()

N, M = 2000, 20
A = sparse(I,N,N) + sprand(N,N,1/N)
invA = inv(Matrix(A))

@testset "Schur complement: " begin
    for i ∈ 1:2000

        y = sprand(N,M,1/sqrt(N*M))

        S = mumps_schur_complement(A,y)

        schur_inds = unique!(sort(y.rowval))

        @test norm(inv(S) - invA[schur_inds,schur_inds]) ≤ sqrt(eps())
    end
end
