struct proyecto_final
    #Variables necesarias
    f::Function             #Funcion a minimizar
    x0::Vector{Float64}     #Valor inicial
    tol::Float64            #Tolerancia
    maxit::Int64            #Maximo numero de iteraciones
    met::String             #Metodo que se va a usar para minimizar
    res::Vector{Float64}    #Valor minimizado

    #Constructor
    function proyecto_final(f::Function, x0::Array; tol::Float64=1e-4,
        maxit::Int64=10000, met::String="NEWT-H")

        # Se hace el metodo seleccionado
        if met=="BFGS"
            res=BFGS(f, x0, tol, maxit)
        elseif met=="NEWT-H"
            res=newton_hessiana(f, x0, tol, maxit)
        elseif met=="NEWT"
            res=newton(f, x0, tol, maxit)
        elseif met=="LINE"
            res=line_search(f, x0, tol, maxit)
        else
            # Si no es un metodo valido manda un error
            error(string(met,": no es un metodo valido\n",
                        "Los metodos validos son: ", metodos))
        end
        # Construye el struct
        new(f, x0, tol, maxit, met, res)
    end


    #=
    Metodos principales
    =#
    function BFGS(f, x0, tol, maxit)
        #=
        TO DO
        =#
        return [NaN, NaN]
    end
    function newton_hessiana(f, x0, tol, maxit)
        #=
        TO DO
        =#
        return [NaN, NaN]
    end
    function newton(f, x0, tol, maxit)
        #=
        TO DO
        =#
        return [NaN, NaN]
    end
    function line_search(f, x0, tol, maxit)
        #=
        TO DO
        =#
        return [NaN, NaN]
    end

    #=
    Metodos auxiliares
    =#

end
