using LinearAlgebra
using CSV
using DataFrames
using ForwardDiff
using Plots
Pkg.add("Plots")

#Pkg.add("ForwardDiff")
struct proyecto_final
    # Variables necesarias
    f::Function             #Funcion a minimizar
	gf::Function 			#Derivada de f
    x0::Vector{Float64}     #Valor inicial
    tol::Float64            #Tolerancia
    maxit::Number           #Maximo numero de iteraciones
    met::String             #Metodo que se va a usar para minimizar
    res::Vector{ComplexF64}    #Valor minimizado
	pasos::Array{Union{Array{Float64,1}, Nothing}}	#Puntos por los que paso el algoritmo
    #Constructor
    function proyecto_final(f::Function, x0::Array; tol::Float64=1e-4,
        maxit::Number=1e6, met::String="NEWT-H", a=1.0, gf=NaN, c1=1e-4,
		c2=0.9, p=2.0)
        # Se hace el metodo seleccionado
        if met=="BFGS"
			#Creo que deberia de incluir c1,c2 y p
            des=BFGS(NaN, c1, c2, p)
        elseif met=="NEWT-H"
            des=Newton_H(a)
        elseif met=="NEWT"
			#Creo que deberia de incluir c1,c2 y p
            des=Newton()
        elseif met=="LINE"
			#Creo que deberia de incluir c1,c2 y p
            des=Line_Search(c1, c2, p)
        else
            # Si no es un metodo valido manda un error
            error(string(met,": no es un metodo valido\n",
                        "Los metodos validos son: ", ["BFGS","NEWT-H","NEWT",
						"LINE"]))
        end

		res, p=descent(f, x0, des; tol=tol, maxit=maxit, Gf=gf)
        # Construye el struct
		if isnan(gf)
        	new(f, x -> grad(f,x), x0, tol, maxit, met, res, p)
		else
			new(f, gf, x0, tol, maxit, met, res, p)
		end
		return res
    end

end


################################################################################
#################### Declaracion de tipos de ###################################
########################## Decenso #############################################
################################################################################
abstract type DescentMethod end

#=
## BFGS
=#
mutable struct BFGS  <: DescentMethod
    Q
	c1
	c2
	p
end
function init!(D::BFGS, x, gf)
    m = length(x)
    D.Q = Matrix(1.0I, m, m) #Matriz identidad de mxm
    return D
end
function step!(D::BFGS, f, gf, x, gx, hx)
    Q = D.Q
    xk = line_step_size(f, x, -Q*gx, D) #TODO
    gk = gf(xk)
    d = xk - x
    y = gk - gx
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
	c1
	c2
	p
end
function init!(D::Line_Search, x, gf)
    return D
end
function step!(D::Line_Search, f, gf, x, gx, Hx)
	#println(gx, grad(f,x))
	xk=line_step_size(f, x, -gx, D)
	#println(xk,"\n")
	#sleep(1)
	return xk, gf(xk)
end

function line_step_size(f, x, d, des::DescentMethod)
	c1=des.c1; c2=des.c2; p=des.p
	g = al -> f(x + al*d)
	dg = al -> derivate(g, al)
	gdg(a)= g(a),dg(a)
	al = minimize(g)
	if isnan(al)
		println(x)
	end
	#println(al, -d, x)

	return x + al*d
end

function minimize(f; c1=1e-4,c2=0.9,p=2)
	a_0=0.0; a_imin=a_0; a_i=1e-3; a_max=65536.0
	i=0
	while a_i<a_max
		if (f(a_i)>f(0)+c1*a_i*derivate(f,0)) || (f(a_i)>=f(a_imin) && i>0)
			return zoom(a_imin, a_i, f, c1, c2)
		end

		if abs(derivate(f,a_i))<=-c2*derivate(f,0)
			return a_i
		end

		if derivate(f,a_i)>=0
			return zoom(a_i, a_imin, f, c1, c2)
		end

		a_imin=a_i
		a_i *= p
		i+=1
	end
	return 1
end

function zoom(al, ah, f, c1, c2)

	ak=NaN

	for i in 0:10
		if al < ah
			ak=interpolate(al, ah, f)
		else
			ak=interpolate(ah, al, f)
		end

		if isnan(ak)
			println(String("\n",i,"    aiuda"))
		end
		if (f(ak)>f(0)+c1*ak*derivate(f,0)) || (f(ak) > f(al))
			ah=ak
		else

			if abs(derivate(f,ak)) <= -c2* derivate(f,0)
				return ak
			end

			if derivate(f, ak) * (ah-al)>=0
				ah=al
			end

			al=ak

		end
	end
	return ak
end



function interpolate(a1, a2, f)
	return a1+(a2-a1)/2
end


################################################################################
####################### Funciones general de ###################################
############################### Decenso ########################################
################################################################################
function descent(f, x0, D::DescentMethod; tol=1e-8, maxit=1e6, Gf=NaN)
	if isnan(Gf)
		gf(x)=grad(f,x) #La derivada de f
	else
		gf=Gf
	end
	opt=false #Cambia a true cuando se encuentra una solucion optima
	D=init!(D,x0,gf) #inicializa el metodo de descenso
	x=copy(x0) #Para no modificar x0
	gx=gf(x); Hx=hess(f,x); it=0
	pasos=Array{Union{Array{Float64,1}, Nothing}}(nothing,trunc(Int, maxit)+1)
	for i in 1:trunc(Int, maxit)
		pasos[i]=x
		x, gx=step!(D,f,gf,x, gx, Hx) #Se da un paso considerando el metodo de descenso
		Hx=hess(f,x)
		if check_optimality(gx,Hx;tol=tol)
			it=i; break
		end
		if i%10000==0
			print(string(i,"...\t", x))
		end
		if isnan(x[1])
			println(i)
			return 0
		end
	end
	if it!=0
		pasos[it+1]=x
		println(string("Se encontro la solucion optima en ", it, " iteraciones"))
	else
		pasos[trunc(Int,maxit)+1]=x
		println("No se encontro la solucion optima, aumente el numero de iteraciones o aumente la tolerancia")
	end
	print(x)
	return x, pasos
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

 	for i in 1:n
        xt1=convert(Array{ComplexF64,1}, x0) #Hacemos una copia y convertimos en arreglo de numeros complejos
        xt1[i]+=h*im #Hacemos el paso complejo (im es i)
        res[i]=imag(f(xt1)) #Extraemos la parte imaginaria de la funcion con paso complejo
		res[i]/=h #Dividimos entre el tamaño del paso
    end
    #return ForwardDiff.gradient(f,x0)
	return res
end
function derivate(f, x; h=1e-20)
	#return ForwardDiff.gradient(f,x)
	return imag(f(x+h*im))/h
end

function hess(f::Function, x0::Array; h::Float64=1e-7)::Array{Float64, 2}
	#Calcula la hessiana de una funcion en un vector, usando paso complejo y paso centrado
	#Algoritmo sacado de:
	# Yi Cao (2020). Complex step Hessian (https://www.mathworks.com/matlabcentral/fileexchange/18177-complex-step-hessian), MATLAB Central File Exchange. Retrieved October 2, 2020.

	n = length(x0)
	H=zeros(n,n)
	h2=h^2
	for i in 1:n
	    	x1=convert(Array{ComplexF64,1}, copy(x0))
	    	x1[i] += h*im #Se hace el paso complejo en el valor i
 			for j=i:n
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

	 #return ForwardDiff.hessian(f,x)
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

function revisar(x0::Array, a::Number, b::Number; met="NEWT-H", tol::Float64=1e-10, maxit::Int=10000)
	f(x)=(a-x[1])^2+b*(x[2]-x[1]^2)^2
	println("El minimo real es:\t\tf([",a,",",a^2,"])=0")
	p= proyecto_final(f, x0; met=met, tol=tol, maxit=maxit)
	x=p.res
	println("El minimo que se encontro es:\tf([", x[1], ",", x[2], "])=", f(x), "\n\n")
	println("El error absoluto en x es:\t",abs(x[1]-a))
	println("El error relativo en x es:\t",abs((x[1]-a)/a),"\n")
	println("El error absoluto en y es:\t",abs(x[2]-a^2))
	println("El error relativo en y es:\t",abs((x[2]-a^2)/a^2))
	return p
end
#revisar([rand(1:10),rand(1:100)], rand(1:10), rand(1:1000); tol=1e-10, maxit=10000, met="NEWT-H")

################################################################################
############################# Funcion de las ###################################
################################# camaras ######################################
################################################################################


function costo_camaras(x, x0)
	res=0
	n= convert(Int64,floor(length(x0)/2))
	m = convert(Int64,floor(length(x)/2))
	for j in 1:m
		for i in 1:m
			if i!=j
				c = ((x[j]-x[i])^2+(x[j+m]-x[i+m])^2)
				if real(c) <  1e-16    #Cota de distancia minima entre camaras
					res += 1000
				else
					res += 1/real(c)
				end
			end
		end
	end
	for i in 1:n       #n crimenes x0
		for j in 1:m   #m camaras x
			res+= (x[j]-x0[i])^2+(x[j+m]-x0[i+n])^2
		end
	end
	#print(res)
	print('\n')
	return res
end


function rand_geo_array(mina, maxa, mino, maxo, s)
	lat=(maxa-mina)*rand(s).+mina
	lon=(maxo-mino)*rand(s).+mino
	return cat(lat,lon, dims=1)
end

function mejores_camaras(dir, n; tol::Float64=1e-4,
	maxit::Number=1e6, met::String="NEWT-H", a=1.0, gf=NaN, c1=1e-4,
	c2=0.9, p=2.0)
	df = CSV.read(dir, DataFrame)
	#y0=rand_geo_array(minimum(df.lat),maximum(df.lat),minimum(df.long),maximum(df.long), 10000)
	x = convert(Array{Float64,1},df.lat)
	y = convert(Array{Float64,1},df.long)

	xmin = minimum(x)
	xmax = maximum(x)
	ymin = minimum(y)
	ymax = maximum(y)

	#pasamos todo al intervalo [0,1]
	for i in 1:length(x)
	    x[i] = (1/(xmax - xmin))*(x[i] - xmin)
	    y[i] = (1/(ymax - ymin))*(y[i] - ymin)
	end


	y0 = cat(x[1:200],y[1:200],dims = 1)
	x = zeros(2*n)
	gr()
	#Plots.plot(y0[1:50],y0[51:100])
	#y0 = [0,0,1,1,0,1,0,1]
	f(x)=costo_camaras(x,y0);

	return proyecto_final(f, x; tol=tol,maxit=maxit, met=met, a=a, gf=gf, c1=c1,c2=c2, p=p), y0
end


result= mejores_camaras("/home/fran/Documents/Aplicado/alumnos/danjinich/proyecto_final/crime_data.csv", 10, maxit=100000, met="BFGS")
Plots.scatter!(result[1][1:4],result[1][5:8])
Plots.scatter(result[2][1:200],result[2][201:400])
plt = Plots.scatter!(result[1][1:10],result[1][11:20], marker = :s )
Plots.savefig(plt,"200-10-3aParam.png")
#Plots.scatter!(result[2][1:4],result[2][5:8])


#rand_geo_array(1, 2, 3, 4, 5)

Plots.save
