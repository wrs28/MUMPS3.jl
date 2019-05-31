using MUMPS3
using MPI, LinearAlgebra, SparseArrays

MPI.Initialized() ? nothing : MPI.Init()

N, M = 2000, 20

@testset "Basic solve: " begin
for i ∈ 1:2000
        A = sparse(I,N,N) + sprand(N,N,1/N)
        y = sprand(N,M,1/sqrt(N*M))

        x = mumps_solve(A,y)

        @test norm(A*x-y) ≤ sqrt(eps())
    end
end
