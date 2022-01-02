import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint

'''
Hay dos formas para resolver edos de esa libreria:
 (1) odeint -> más vieja
 (2) solve_ivp -> + nueva y cn más opciones.
'''
#################

'''
obs
 > np.linspace -> Return evenly spaced numbers over a specified interval.
    
    
> Pendiente
    - Class Parameters: Leida
    
'''


def f(y, t, paras):
    """
    Your system of differential equations
    """
    x1 = y[0] #tengo esto en rhs para solve_ivp
    x2 = y[1] #variables
    x3 = y[2]

    # En el momento que se captura una excepción dentro de try el flujo del
    # programa salta inmediatamente al bloque de una de las sentencias except
    try:
        k0 = paras['k0'].value #constantes == parámetros :)
        k1 = paras['k1'].value

    except KeyError: # uso incorrecto o inválido de llaves (keys) en diccionarios
        k0, k1 = paras

    # The model equations -> en mi caso las ODE's
    f0 = -k0 * x1
    f1 = k0 * x1 - k1 * x2
    f2 = k1 * x2

    return [f0, f1, f2]



def g(t, x0, paras):
    #ese 'x0' en la documentación aparece como 'y0', y es xq acá x es x(t).
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0 [para UN t.]
    """
    #print(f't en odeint: {t}')
    # ODE INT TRABAJA CN EL t cmo array, mientras q solve_ivp NO
    x = odeint(f, x0, t, args=(paras,))
    print(f'resulrado odeint: {x}') # array de 3, xq "f0, f1, f2", para mi es algo cmo T,L... pero solo querré L.
    print(f'x.shape: {x.shape}')
    # "args=(paras,) es xq entremedio ese keyword argument va a ser llamado como *args y se va
    # a desempaquetar entregando a f solo 'paras' "

    #AQUÍ USAN ODEINT, PERO AL MENOS SOLVE_IVP HACE LO SGTE:
        ## solve_ivp integra, i.e. obtiene sols en (t,t+1) y luego evalua eso en t_eval.
        #Devuelve 1. sol.y -> array [[T][L][M][I]]) , 2. sol.t -> array [t] -> i.e sols en ese t.
    return x


def residual(paras, t, data):
    # data es algo q tgo q saber y se entrega para q sea comparada cn fitted data.
    """
    compute the residual (¿es la diferencia?) between actual data and fitted data
    """
    x0 = paras['x10'].value, paras['x20'].value, paras['x30'].value
    # '.value' es un atributo de objetos Parameters
    model = g(t, x0, paras)
    print(f'modelo : {model}')

    # you only have data for one of your variables

    x2_model = model[:, 1] # esto arroja un np.array (arreglo) - class 'numpy.ndarray'
    print(f'x2_model : {x2_model}')
    # ? - entiendo cmo fxna, pero xq hace eso?
    # Esto recorre cada elemento ":" del array y va extrayendo el elemento en pos 1 ",1"

    return (x2_model - data).ravel()
        # ? -> Qué es ravel()? -> Parece calcular el residuo entre actual and fitted data


# initial conditions
x10 = 5.
x20 = 0
x30 = 0
y0 = [x10, x20, x30]


# measured data -> data to be fitted
t_measured = np.linspace(0, 9, 12)
    # > np.linspace -> Return evenly spaced numbers over a specified interval.
x2_measured = np.array([0.000, 0.416, 0.489, 0.595, 0.506, 0.493, 0.458, 0.394, 0.335, 0.309, 0.404, 0.500])
#print(type(t_measured))
#print(type(x2_measured))

plt.figure()
plt.scatter(t_measured, x2_measured, marker='o', color='b', label='measured data', s=75)



# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()
    #Guarda los parámetros y luego lo llamas cmo un dict
params.add('x10', value=x10, vary=False) # FIXED PARAMETER -> vary = False
params.add('x20', value=x20, vary=False)
params.add('x30', value=x30, vary=False)
params.add('k0', value=0.2, min=0.0001, max=2.) #LE AGREGO CONSTRAINTS AL VALOR DEL PARÁMETRO
params.add('k1', value=0.3, min=0.0001, max=2.)



# fit model
result = minimize(residual, params, args=(t_measured, x2_measured), method='leastsq')  # leastsq nelder
print(f't_measured: {t_measured}')
print(f't_measured: {type(t_measured)}')

    # obs: t_measured & x2_measured es la DATA q yo tgo

# minimize -> takes an objetive fx, that calculates the array to be minimized
    # returns -> MinimizerResult

'''
A la clase 'minimize' (recordar q hay otra q se llama Minimize)
le entrego el residuo a minimizar, los parámetros a definir y la data q tgo. 

Así tmbn le tengo que pedir cn q MÉTODO ENCONTRARÁ los valores de los parámetros q busco definir
'''

# check results of the fit -> ocupo 'result' q se obtuvo ocupando clase 'minimize'
    #OJO-> ahora recién llamó a 'g'
#           -> 'g' = Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
#                    [para algún t]

data_fitted = g(np.linspace(0., 9., 100), y0, result.params)
        # obs: '9.' es un float

# plot fitted data
plt.plot(np.linspace(0., 9., 100), data_fitted[:, 1], '-', linewidth=2, color='red', label='fitted data')
    #  > np.linspace -> Return evenly spaced numbers over a specified interval.
    #  data_fitted[:, 1] -> arroja np.array (arreglo)
       # Eso recorre cada elemento ":" del array y va extrayendo el elemento en pos 1 ",1"

plt.legend()
plt.xlim([0, max(t_measured)])
plt.ylim([0, 1.1 * max(data_fitted[:, 1])])

# display fitted statistics
report_fit(result) # arroja un printeable q me describe muchas cosas sobre el fit

#plt.show()