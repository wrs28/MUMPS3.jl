module MUMPS_wrs

using MPI,
LinearAlgebra,
SparseArrays

export invoke_mumps!

include("lib_location.jl")
include("mumps_types.jl")
include("mumps_struc.jl")
include("interface.jl")
include("convenience_wrappers.jl")

function invoke_mumps!(mumps::Mumps{TR,TC}) where {TR,TC}
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
    @eval ccall(($(string(cfun)),$(mumps_lib)),
        Cvoid, (Ref{Mumps{$TR,$TC}},), $mumps)
    return nothing
end

end
