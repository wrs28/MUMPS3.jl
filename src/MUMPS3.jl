"""
    module MUMPS3

Both low-level interface with MUMPS 5.1.2 parallel direct solver C-library
as well as convenient wrappers for some common uses for MUMPS.

The central work is done by the `Mumps` struct, which mirrors the
internal structure used in MUMPS. Manipulations can be done directly
on this object and then passed to Mumps via the function `invoke_mumps!`
This mode of operation gives the user complete control as described
in the MUMPS manual, though it exposes unsafe operations, to be cautioned.

More convenient are the use of the functions `mumps_solve`, `mumps_factorize`,
`mumps_det`, `mumps_schur`, and `mumps_select_inv`, which all have
mutating counterparts (such as `mumps_solve!`). These can take matrices
and right hand sides directly, so, for example, the equation `A*x=y`, solved
in Base by `x=A\\y` or `LinearAlbegra.ldiv!(x,A,y)`, can be solved in MUMPS3
as `x=mumps_solve(A,y)`, or `mumps_solve!(x,A,y)`.

The package also extends Base.\\, LinearAlgebra.ldiv! and LinearAlgebra.inv to
work with mumps objects. Finally, the function `mumps_inv(A)` returns a Linear Map
that has the same action on vectors as A⁻¹. Think of it as the counterpart
to an UMFPACK LU factorization object used in left division.

Note, unless working with the low-level interace, I discourage setting the JOB
parameter manually, as this can lead to unsafe operation.

The goal is to give the advanced user low-level access to MUMPS, while simultaneously
giving the ordinary user safe functions that grant access to most of what
MUMPS has to offer.
"""
module MUMPS3

using MPI,
LinearAlgebra,
LinearMaps,
SparseArrays

export invoke_mumps!

include("lib_location.jl")
include("mumps_types.jl")
include("mumps_struc.jl")
include("interface.jl")
include("utilities.jl")
include("convenience_wrappers.jl")
include("icntl_alibis.jl")
include("printing.jl")

end
