using MUMPS3,MPI,LinearAlgebra,SparseArrays

MPI.Initialized() ? nothing : MPI.Init()

N, M = 1000, 10
A = sparse(I,N,N) + sprand(N,N,1/N)
y = sprand(N,M,1/sqrt(N*M))

x = mumps_solve(A,y)
norm(A*x-y)
