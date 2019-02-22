# MUMPS3.jl

The MUMPS3.jl package provides a Julia interface with the MUMPS 5.1.2 parallel direct solver (http://mumps.enseeiht.fr).

This package *does not* come with a distribution of MUMPS, and it is up to the user to provide a working MUMPS library.


## Installation

#### Installing MUMPS3.jl

The package is installed by entering the Pkg environment by typing `]`, followed
by `add git@github.com:wrs28/MUMPS3.jl.git`, which looks like this:
````JULIA
(v1.1) pkg> add git@github.com:wrs28/MUMPS3.jl.git
````

Alternatively, it can be installed through the Pkg package (first call `using Pkg`) via `Pkg.add("git@github.com:wrs28/MUMPS3.jl.git")`.

MUMPS3 will need to be told where the MUMPS library is via the environment variable
`ENV["MUMPS_PREFIX"]`, which defaults to
````JULIA
ENV["MUMPS_PREFIX"] = /usr/local/opt/brewsci-mumps
````
This must be set before each time before loading the package. I recommend putting
it in your startup.jl file:
````JULIA
push!(ENV,"MUMPS_PREFIX"=>/path/to/your/mumps/directory)
````

#### Installing MUMPS

This can be a bit trick. The source code can be downloaded [here](http://mumps.enseeiht.fr),
but compiling and linking into a dynamic library is awkward at best.

On Mac OS, there is an easy alternative from Homebrew, and detailed instructions
can be found [here](https://github.com/JuliaSmoothOptimizers/MUMPS.jl/blob/master/README.md).
In short, the calls
````SHELL
$ brew tap brewsci/num
$ brew install brewsci-mumps
````
should be sufficient for installing mumps and its dependencies.

## Basic Usage

To load the package, simply call `using MUMPS3`. Additionally you will need to load
MPI.jl by calling `using MPI`.

Before any calls to MUMPS3, you must initialize the MPI environment by calling
`MPI.Init()`. If working in interactive mode, to avoid multiples `MPI.Init` calls, and to ensure proper finalization of MPI, I recommend putting
````JULIA
MPI.Initialized() ? nothing : MPI.Init()
MPI.finalize_atexit()
````
at the top of your code.

Now you are poised to begin.

There are five high-level functions that use the [MUMPS library](http://mumps.enseeiht.fr): `mumps_solve`, `mumps_factorize`, `mumps_det`, `mumps_schur`, `mumps_select_inv`. The first three are self-explanatory, and last two 
