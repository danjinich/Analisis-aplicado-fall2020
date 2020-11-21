using LinearAlgebra

mutable struct proyecto_final
    #Variables necesarias
    f::Function             #Funcion a minimizar
	gf::Function 			#Derivada de f
    x0::Vector{Float64}     #Valor inicial
    tol::Float64            #Tolerancia
    maxit::Int64            #Maximo numero de iteraciones
    met::String             #Metodo que se va a usar para minimizar
    res::Vector{Float64}    #Valor minimizado

    #Constructor
    function proyecto_final(f::Function, x0::Array; tol::Float64=1e-4,
        maxit::Int64=10000, met::String="NEWT-H", a=1.0, gf=NaN)

        # Se hace el metodo seleccionado
        if met=="BFGS"
            des=BFGS(NaN)
        elseif met=="NEWT-H"
            des=Newton_H(a)
        elseif met=="NEWT"
            des=Newton()
        elseif met=="LINE"
            des=Line_Search()
        elseif met=="C-GRAD"
            des=ConjugateGradient(NaN,NaN)
		elseif met=="GRAD"
			des=GradientDescent(a)
        else
            # Si no es un metodo valido manda un error
            error(string(met,": no es un metodo valido\n",
                        "Los metodos validos son: ", ["BFGS","NEWT-H","NEWT",
						"LINE", "C-GRAD", "GRAD"]))
        end

		res=descent(f, x0, des; tol=tol, maxit=maxit, Gf=gf)
        # Construye el struct
		if isnan(gf)
        	new(f, x -> grad(f,x), x0, tol, maxit, met, res)
		else
			new(f, gf, x0, tol, maxit, met, res)
		end
    end
end


abstract type DescentMethod end

#=
## BFGS
=#
mutable struct BFGS  <: DescentMethod
    Q
end
function init!(D::BFGS, x, gf)
    m = length(x)
    D.Q = Matrix(1.0I, m, m) #Matriz identidad de mxm
    return D
end
function step!(D::BFGS, f, gf, x, gx, hx)
    Q, g = D.Q, gx
    xk = line_search(f, x, -Q*g)
    gk = gf(xk)
    d = xk - x
    y = gk - g
    Q[:] = Q - (d*y'*Q + Q*y*d')/(d'*y) +
        (1 + (y'*Q*y)/(d'*y))[1]*(d*d')/(d'*y)
    return xk, gk
end

#=
## Metodo de Newton
=#
mutable struct Newton  <: DescentMethod
end
function init!(D::Newton, x, gf)
    return D
end
function step!(D::Newton, f, gf, x, gx, Hx)
	g = Hx \ gx
	xk=x-g
	return xk, gf(xk)
end

#=
## Gradiente
=#

struct GradientDescent <: DescentMethod
	a
end
function init!(D::GradientDescent, x, gf)
    return D
end
function step!(D::Line_Search, f, gf, x, gx, Hx)
	xk=x-D.a*g
	return xk, gf(xk)
end

#=
## Gradiente conjugada
=#
mutable struct ConjugateGradient <: DescentMethod
	d
	g
end
function init!(M::ConjugateGradient, x, gf)
	M.g = gf(x, gf)
	M.d = -M.g
	return M
end
function step!(M::ConjugateGradient, f, gf, x, gx, Hx)
	d, g = M.d, M.g
	b = max(0, dot(gx, gx-g)/(g⋅g))
	dk = -gx + b*d
	xk = line_search(f, x, dk)
	M.d, M.g = dk, gx
	return xk, gf(xk)
end


#=
## Newton con modificacion a la Hessiana
=#
mutable struct Newton_H  <: DescentMethod
	a
end
function init!(D::Newton_H, x, gf)
	return D
end
function step!(D::Newton_H, f, gf, x, gx, Hx)
	B=add_identity(Hx)
	p=B \ gx*(-1)
	D.a=backtracking_line_search(f,x,p, gx; a=D.a)
	xk = x + D.a*p
	return xk, gf(xk)
end

function backtracking_line_search(f, x, d, gx; a=1.0, p=0.5, c=1e-4)
	#Habia que hacerlo
	y = f(x)
	g = gx
	while f(x + a * d) > y+c*a*dot(g,d)
		a *= p
	end
	return a
end

function add_identity(A; b=1e-4)
	#Metodo para convertir en definida positiva una matriz
	#Encuentra una t tal que A+I*t es definida positiva
	n=size(A)[1]; Bk=copy(A); i=Matrix{Float64}(I, n, n)
	if (minimum(diag(A))>0)
		t=0
	else
		t=-minimum(diag(A))+b
	end
	while (!is_pos_def(Bk))
		t=max(2*t,b) #Si no funciona crecemos y volvemos a tratar
		Bk=A+i*t
	end
	return Bk
end

#=
## Line search
=#
mutable struct Line_Search  <: DescentMethod
	#=
	TODO
	=#
end
function init!(D::Line_Search, x, gf)
	#=
	TODO
	=#
    return D
end
function step!(D::Line_Search, f, gf, x, gx, Hx)
	#=
	TODO
	=#
	return xk, gf(xk)
end




function descent(f, x0, D::DescentMethod; tol=1e-8, maxit=1e6, Gf=NaN)
	if isnan(Gf)
		gf(x)=grad(f,x) #La derivada de f
	else
		gf=Gf
	end
	opt=false #Cambia a true cuando se encuentra una solucion optima
	D=init!(D,x0,gf) #inicializa el metodo de descenso
	x=copy(x0) #Para no modificar x0
	gx=gf(x); Hx=Hess(f,x)
	for i in 1:maxit
		x, gx=step!(D,f,gf,x, gx, Hx)) #Se da un paso considerando el metodo de descenso
		Hx=hess(f,x)
		if check_optimality(gx,Hx;tol=tol)
			it=i; opt=true; break
		end
	end
	if opt
		println(string("Se encontro la solucion optima en ", it, " iteraciones"))
	else
		println("No se encontro la solucion optima, aumente el numero de iteraciones o aumente la tolerancia")
	end
	return x
end






################################################################################
#################### Funciones necesarias de ###################################
######################## Algebra lineal ########################################
################################################################################

function is_pos_semi_def(A::Array{Float64, 2})::Bool
	#Checa si es semidefinida positiva calculando los eigenvalores y checando que sean >= 0
	return all(x->x>=0,eigvals(A))
end

function is_pos_def(A::Array{Float64, 2})::Bool
	#Checa si es definida positiva calculando los eigenvalores y checando que sean > 0
	return all(x->x>0,eigvals(A))
end

function grad(f::Function, x0::Array; h::Float64=1e-20)::Array{Float64,1}
    #Encuentra el gradiente de vectores haciendo un paso complejo
    n=length(x0)
    res=Array{Float64}(undef, n) #Arreglo vacio

    Threads.@threads for i in 1:n
        xt1=convert(Array{ComplexF64,1}, x0) #Hacemos una copia y convertimos en arreglo de numeros complejos
        xt1[i]+=h*im #Hacemos el paso complejo (im es i)
        res[i]=imag(f(xt1)) #Extraemos la parte imaginaria de la funcion con paso complejo
        res[i]/=h #Dividimos entre el tamaño del paso
    end
    return res
end

function hess(f::Function, x0::Array; h::Float64=1e-7)::Array{Float64, 2}
	#Calcula la hessiana de una funcion en un vector, usando paso complejo y paso centrado
	#Algoritmo sacado de:
	# Yi Cao (2020). Complex step Hessian (https://www.mathworks.com/matlabcentral/fileexchange/18177-complex-step-hessian), MATLAB Central File Exchange. Retrieved October 2, 2020.
	n = length(x0)
	H=zeros(n,n)
	h2=h^2
	Threads.@threads for i in 1:n
	    	x1=convert(Array{ComplexF64,1}, copy(x0))
	    	x1[i] += h*im #Se hace el paso complejo en el valor i
	    	Threads.@threads for j=i:n
	    		x2=copy(x1)
	    		x2[j] += h #Se hace un paso real hacia delante en el valor j
	    		u1=f(x2)
	    		x2[j] = x1[j] - h #Se hace un paso real hacia atras en el valor j
	    		u2 = f(x2)
	    		H[i,j] = imag(u1-u2)/(2*h2) #Se extrae la diferencia de la parte imaginaria y se divide entre 2*h^2
	    		H[j,i]=H[i,j] #Ya que es simetrica la matriz
    		end #for
   	 end # for
    return H
end

function check_optimality(grad::Array{Float64,1}, hess::Array{Float64, 2}; tol::Float64=1e-20)::Bool
    #Checa optimalidad
    if all(x->abs(x)<=tol,grad) #El gradiente es menor a la tolerancia
        return is_pos_semi_def(hess) #La Hessiana es semidefinida positiva
    end
    return false
end


################################################################################
############################# Funciones para ###################################
############################### optimizar ######################################
################################################################################
function rosenbrock(x0::Array; a::Number=1.0, b::Number=100.0)::Number
	return (a-x0[1])^2+b*(x0[2]-x0[1]^2)^2
end
