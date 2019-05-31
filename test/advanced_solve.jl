using MUMPS3
using MPI, LinearAlgebra, SparseArrays

MPI.Initialized() ? nothing : MPI.Init()

N, M = 2000, 20

@testset "Advanced solve: " begin
    for i ∈ 1:2000
        A = sparse(1.0im*I,N,N) + sprand(N,N,1/N)
        y = sprand(N,M,1/sqrt(N*M))

        m = Mumps(A,y)
        toggle_display!(m)

        x = mumps_solve(m)
        @test norm(A*x-y) ≤ sqrt(eps())
    end
end
