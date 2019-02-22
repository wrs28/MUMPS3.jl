export display_icntl

Base.show(io::IO,mumps::Mumps) = show(mumps.mumpsc)

function Base.show(io::IO,mumpsc::MumpsC{TC,TR}) where {TC,TR}
    print("Mumps{$TC,$TR}: ")
    if TC<:Float32
        println("single precision real")
        lib = "smumps"
    elseif TC<:Float64
        println("double precision real")
        lib = "dmumps"
    elseif TC<:ComplexF32
        println("single precision complex")
        lib = "cmumps"
    elseif TC<:ComplexF64
        println("double precision complex")
        lib = "zmumps"
    end
    println("lib: ", lib)
    print("job: ", mumpsc.job, " ")
    if mumpsc.job==-2
        println("terminate")
    elseif mumpsc.job==-1
        println("initialize")
    elseif mumpsc.job==1
        println("analyze")
    elseif mumpsc.job==2
        println("factorize")
    elseif mumpsc.job==3
        println("solve")
    elseif mumpsc.job==4
        println("analyze + factorize")
    elseif mumpsc.job==5
        println("factorize + solve")
    elseif mumpsc.job==6
        println("analyze + factorize + solve")
    else
        println("unrecognized")
    end
    print("sym: ", mumpsc.sym)
    if mumpsc.sym==1
        println(" symmetric pos def")
    elseif mumpsc.sym==2
        println(" symmetric")
    else
        println(" unsymmetric")
    end
    print("par: ", mumpsc.par)
    mumpsc.par==0 ? println(" host not worker") : println(" host is worker ")
    print("matrix A: ")
    if has_matrix(mumpsc)
        print("$(mumpsc.n)×$(mumpsc.n) ")
        if is_matrix_assembled(mumpsc)
            println("sparse matrix, with $(mumpsc.nnz) nonzero elements")
        else
            print("elemental matrix with $(mumpsc.nelt) element")
            mumpsc.nelt>1 ? println("s") : println()
        end
    else
        println("uninitialized")
    end

    print("rhs B:")
    rhs_type = is_rhs_dense(mumpsc) ? "dense" : "sparse"
    nz_rhs = is_rhs_dense(mumpsc) ? "" : string(",with ",mumpsc.nz_rhs," nonzero elements")
    if has_rhs(mumpsc)
        lrhs = is_rhs_dense(mumpsc) ?  mumpsc.lrhs : mumpsc.n
        nrhs = mumpsc.nrhs
        println(" $lrhs×$nrhs ",rhs_type," matrix", nz_rhs)
    else
        println(" uninitialized")
    end

    println("ICNTL settings summary: ")
    icntl_inds = [4,9,13,19,30,33]
    for i ∈ eachindex(icntl_inds)
        print("\t")
        display_icntl(mumpsc.icntl,icntl_inds[i],mumpsc.icntl[icntl_inds[i]])
    end
end

function Base.show(io::IO,gc_haven::GC_haven)
    vars = (:irn,:jcn,:a,
            :irn_loc,:jcn_loc,:a_loc,
            :eltptr,:eltvar,:a_elt,
            :perm_in,:sym_in,:uns_in,
            :colsca, :rowsca,
            :rhs, :redrhs, :rhs_sparse,
            :sol_loc, :irhs_sparse,
            :irhs_ptr, :isol_loc,
            :pivnul_list, :mapping,
            :listvar_schur, :schur, :wk_user)
    pad = ("::\t",":\t",":\t\t",
            ": ","\t",":\t",
            ":\t","\t",":\t",
            ": ","\t",":\t",
            ":\t",":\t",
            ":\t",":\t",":\t",
            ": ",": ",
            ": ",": ",
            ": ",": ",
            ": ",":\t",": ")
    for i ∈ eachindex(vars)
        println(vars[i],pad[i],isdefined(gc_haven,vars[i]))
    end
end

"""
    display_icntl(mumps)

Show the complete INCTL integer array of `mumps`, with descriptions

See also: [`set_icntl!`](@ref)
"""
display_icntl(mumps::Mumps) = display_icntl(mumps.mumpsc.icntl)
function display_icntl(icntl)
    for i ∈ eachindex(icntl)
        display_icntl(icntl,i,icntl[i])
    end
end
function display_icntl(icntl,i,val)
    automatic = "decided by software"
    print("$i,\t$val\t")
    if i==1
        print("output stream for error messages: ")
        if val≤0
            print("suppressed")
        else
            print("$val")
        end
    elseif i==2
        print("output stream for diagnostics, statistics, warnings: ")
        if val≤0
            print("suppressed")
        else
            print("$val")
        end
    elseif i==3
        print("output stream for global info: ")
        if val≤0
            print("suppressed")
        else
            print("$val")
        end
    elseif i==4
        print("level of printing: ")
        if val≤0
            print("error, warnings, diagnostics suppressed")
        elseif val==1
            print("only error messages")
        elseif val==2
            print("errors, warnings, main statistics")
        elseif val==3
            print("errors, warnings, terse diagnostics")
        else
            print("errors, warnings, info on input, output parameters")
        end
    elseif i==5
        print("matrix input format: ")
        if val==1
            print("elemental")
        else
            print("assembled")
        end
    elseif i==6
        print("permutation and/or scaling: ")
        if val==1
            print("number of diagonal nonzeros is maximized")
        elseif val ∈ [2,3]
            print("smallest diagonal value is maximized")
        elseif val==4
            print("sum of diagonal values is maximized")
        elseif val ∈ [5,6]
            print("product of diagonal values is maximized")
        elseif val==7
            print(automatic)
        else
            print("none")
        end
    elseif i==7
        print("reordering for analysis: ")
        if val==0
            print("Approximate Minimum Degree (AMD)")
        elseif val==1
            print("given by user via PERM_IN (see provide_perm_in)")
        elseif val==2
            print("Approximate Minimum Fill (AMF)")
        elseif val==3
            print("SCOTCH, if installed, else ",automatic)
        elseif val==4
            print("PORD, if installed, else ",automatic)
        elseif val==5
            print("METIS, if installed, else ",automatic)
        elseif val==6
            print("Approximate Minimum Degree with quasi-dense row detection (QAMD)")
        else
            print(automatic)
        end
    elseif i==8
        print("scaling strategy: ")
        if val==-2
            print("computed during analysis")
        elseif val==-1
            print("provided by user in COLSCA and ROWSCA (see provide_)")
        elseif val==1
            print("diagonal, computed during factorization")
        elseif val==3
            print("comlumn, computed during factorization")
        elseif val==4
            print("row and column based on inf-norms, computed during factorization")
        elseif val==7
            print("row and column iterative, computed during factorization")
        elseif val==8
            print("row and column iterative, computed during factorization")
        elseif val==77
            print(automatic," during analysis")
        else
            print("none")
        end
    elseif i==9
        print("transposed: ")
        if val==1
            print("false")
        else
            print("true")
        end
    elseif i==10
        print("iterative refinement: ")
        if val<0
            print("fixed number of iterations")
        elseif val>0
            print("until convergence with max number of iterations")
        else
            print("none")
        end
    elseif i==11
        print("statistics of error analysis: ")
        if val==1
            print("all (including expensive ones)")
        elseif val==2
            print("main (avoid expensive ones)")
        else
            print("none")
        end
    elseif i==12
        print("ordering for symmetric: ")
        if val==1
            print("usual, nothing done")
        elseif val==2
            print("on compressed graph")
        elseif val==3
            print("constrained ordering, only used with AMF (see icntl 7)")
        else
            print(automatic)
        end
    elseif i==13
        print("parallelism of root node: ")
        if val==-1
            print("force splitting")
        elseif val>0
            print("sequential factorization (ScaLAPACK not used) unless num workers > $val")
        else
            print("parallel factorization")
        end
    elseif i==14
        print("percentage increase in estimated working space: $val%")
    elseif i==18
        print("distributed input matrix: ")
        if val==1
            print("structure provided centralized, mumps returns mapping, user provides entries to mapping")
        elseif val==2
            print("structure provided centralized at analysis, entries provided to all workers at factorization")
        elseif val==3
            print("distributed matrix, pattern, entries provided")
        else
            print("centralized")
        end
    elseif i==19
        print("Schur complement: ")
        if val==1
            print("true, Schur complement returned centralized by rows")
        elseif val ∈ [2,3]
            print("true, Schur complement returned distributed by columns")
        else
            print("false, complete factorization")
        end
    elseif i==20
        print("rhs: ")
        0<val<4 ? print("sparse, ") : nothing
        if val==1
            print("sparsity-exploting acceleration of solution ", automatic)
        elseif val==2
            print("sparsity not exploited in solution")
        elseif val==3
            print("sparsity exploited to accelerate solution")
        else
            print("dense")
        end
    elseif i==21
        print("distribution of solution vectors: ")
        if val==1
            print("distributed")
        else
            print("assembled and stored in centralized RHS")
        end
    elseif i==22
        print("out-of-core (OOC) factorization and solve: ")
        if val==0
            print("false")
        elseif val==1
            print("true")
        else
            @warn "not sure this is a valid setting"
        end
    elseif i==23
        print("max size (in MB) of working memory per worker: ")
        if val>0
            print("$val MB")
        else
            print(automatic)
        end
    elseif i==24
        print("null pivot row detection: ")
        if val==1
            print("true")
        else
            print("false. if null pivot present, will result in error INFO(1)=-10")
        end
    elseif i==25
        print("defecient matrix and null space basis: ")
        if val==-1
            print("complete null space basis computed")
        elseif val==0
            print("normal solution phase. if matrix found singular, one possible solution returned")
        else
            print("$val-th vector of null space basis computed")
        end
    elseif i==26
        print("if Schur, solution phase: ")
        if val==1
            print("condense/reduce rhs on Schur")
        elseif val==2
            print("expand Schur solution on complete solution variables")
        else
            print("standard solution")
        end
    elseif i==27
        print("blocking size for multiple rhs: ")
        if val<0
            print(automatic)
        elseif val==0
            print("no blocking, same as 1")
        else
            print("blocksize=min(NRHS,$val)")
        end
    elseif i==28
        print("ordering computation: ")
        if val==1
            print("sequential")
        elseif val==2
            print("parallel")
        else
            print(automatic)
        end
    elseif i==29
        print("parallel ordering tool: ")
        if val==1
            print("PT-SCOTCH, if available")
        elseif val==2
            print("PARMETIS, if available")
        else
            print(automatic)
        end
    elseif i==30
        print("compute entries of A⁻¹: ")
        if val==1
            print("true")
        else
            print("false")
        end
    elseif i==31
        print("discarded factors: ")
        if val==1
            print("all")
        elseif val==2
            print("U, for unsymmetric")
        else
            print("none, except for ooc factorization of unsymmetric")
        end
    elseif i==32
        print("forward elimination of rhs: ")
        if val==1
            print("performed during factorization")
        else
            print("not performed during factorization (standard)")
        end
    elseif i==33
        print("compute determinant: ")
        if val==0
            print("false")
        else
            print("true")
        end
    elseif i==35
        print("BLR: ")
        if val==1
            print("activated")
        else
            print("not activated")
        end
    else
        print("not used")
    end
    print("\n")
    return nothing
end
