#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
from scipy.integrate import solve_ivp
from L_count_temporal import L_count # P. TEMPORAL
'''
Hay dos formas para resolver edos de esa libreria:
 (1) odeint -> más vieja
 (2) solve_ivp -> + nueva y cn más opciones. -> YO OCUPO ESTA
'''

## FLUJO
#Definimos la función que le entrega el lado derecho de cada EDO, de un sistema
# de EDO's de primer orden, a solve_ivp
#(Esto ya lo había hecho en el código ppal, ahora le agregaré manejo de excepciones xq
# si y le pediré q reciba los parámetros a ajustar como objetos de clase Parameters)


## PARÁMETROS OBTENIDOS DE LA LITERATURA (4, fixed):
a = .01  ##[days**-1] #Tumor growth
alpha_T = .139  ##[Gy**-1] #tumor-LQ(linear quadratic) cell kill
beta_T = alpha_T/14.3  ##[Gy**-2]
f = .033 ##[days**-1]  #lymphocyte decay rate
r = 0.14 ##[days**-1] #half life of 5 days



## # rhs = right hand side of the ode
def rhs(t, y, parametros): # LISTA
    """
    Input Ode: Fx entrega odes a solve_ivp en el formato pedido.
    # parámetros = a ajustar como objetos de clase Parameters
    """

    T_count, L_count, M_count, I_count = y      # variables
    "obs: C.I.'s vienen cmo un array"

    try:                                        # parámetros a ajustar
        omega_1 = parametros['omega_1'].value
        omega_2 = parametros['omega_2'].value
        omega_3 = parametros['omega_3'].value
        g = parametros['g'].value
        s = parametros['s'].value
        #alpha_T = parametros['alpha_T'].value
        #alpha_L = parametros['alpha_L'].value
            #obs: alpha_T & alpha_L aparecen solo en la parte de radiación
            # así q no los ocuparé por ahora (?)

    except KeyError: # uso incorrecto o inválido de llaves (keys) en diccionarios
        omega_1, omega_2, omega_3, g, s, alpha_T, alpha_L = parametros


    # ODE's -> SIN COMPONENTE RADIACIÓN
    T_dot = a * T_count - omega_1 * ( T_count/(g+T_count+M_count) ) * L_count

    L_dot = omega_2 * ( (T_count + M_count)/(g+T_count+M_count) ) * L_count \
            + omega_3 * ( I_count/(g+I_count) )* L_count + s - (f * L_count)

    M_dot = a * M_count - omega_1 * ( M_count/(g+T_count+M_count) ) * L_count

    I_dot = - r * I_count


    dydt = np.array([T_dot, L_dot, M_dot, I_dot])

    return dydt

# P. Incluir efectos radiación sobre conteo de células.


## Solución de la EDO para UN t
def sol_ode_en_t(t, y0, parametros): # y(t)
    """
    - Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0 [para UN t.]
    - ocuparemos solve_ivp
    """

    # obs:
    # ANTES -> y0 = np.array([T,L,M,I]) #Condiciones iniciales
    # -> AHORA SE LA ENTREGAMOS A LA FX 'sol_ode_en_t' CMO ARGUMENTO

    sol = solve_ivp(rhs, (t,t+1), y0, t_eval = np.array([t+1]), max_step = 0.001,
                    args=(parametros,))
        #LO NUEVO -> 'args=(parametros,)'
    #t_eval = t, q me entrega, en q se evalua edo (puede evaluar en más puntos, xq eso lo define
    # inteligentemente solve_ivp internamente, solo q no me los entrega); ese t tiene q estar
    # dentro d (t,t+1) solve_ivp integra, i.e. obtiene sols en (t,t+1) y luego evalua eso
    # en t_eval.
    #Devuelve sol.y -> array [[T][L][M][I]]) , sol.t -> array [t]

    return sol



## Función que calcula el residuo entre la data q tgo cn la fitteada sg el método escogido
def residuo(parametros, t, data):
    """
    Compute the residual (¿cmo se obtiene este residuo?) between actual data and fitted data
    """

    # Le entrego los valores iniciales como objeto Parameters fijo (fixed value)
    y0 = (parametros['omega_1'].value, parametros['omega_2'].value, parametros['omega_3'].value,
          parametros['g'].value, parametros['s'].value)

    # Defino el modelo matemático de los datos (antes: f(x) = x**2 por ejemplo. ahora: ODE's)
    model = sol_ode_en_t(t, y0, parametros)
    # Esto va a arrojar un arreglo (np.array)
    #P. # y por una razón q aún no entiendo (esto lo extraje de "Ej minimizer & Odes.py") vamos a:
    # Recorrer cada elemento ":" del array y extraer de eso el elemento en pos 1 ",1"
    y_model = model[:, 1]

    return (y_model - data).ravel()
    # Parece calcular el residuo entre actual and fitted data}
         # ? -> Debe ser un método de alguna clase o algo q no identifico bn, será un built in?



## LISTAS LAS FXNES. AHORA LA DATA Y PLOTTEOS


# 1. Condiciones iniciales
    # T:(primary) Tumor cells
    # L: Linfocitos
    # M: Metastatic Tumor Cells
    # I: Inactivated tumor cells realising antigens

T0 = 1.07 * 10 ** 11
L0 = 5.61 * 10 ** 9
M0 = 1.07 * 10 ** 8    # Patient with metastasis
I0 = 0                 # T deaths = 0 at the beginning
y0 = np.array([T0, L0, M0, I0]) # ocupo array xq solve ivp pide un array para las C.I.


# 2. measured data(data to be fitted) + plot de data
    # Data centro de cancer UC -> PENDIENTE

    # temporal: datos inventados.
t_medido = np.linspace(0, 100, 101) # 101 números entre 0 y 100 igualmente espaciados.
y_medido_L = L_count # en realidad esto es lo fitted.


#plt.figure() -> ? NO SÉ K HACE ESTO
plt.scatter(t_medido, y_medido_L, marker='o', color='b', label='measured data', s=30)
    #plt.scatter -> A scatter plot of y vs. x with varying marker size and/or color.
    # Se hace para ver en el plot la data REAL junto cn la fitter q estamos buscando




#3. set parameters including bounds; you can also fix parameters (use vary=False)
v_parametros = Parameters() #variable parametros, así no se cambia su nombre en las fxnes.
    #Guarda los parámetros y luego lo llamas cmo un dict

# añadimos las C.I. como parámetros fijos (FIXED)
v_parametros.add('T0', value = T0, vary = False) # FIXED PARAMETER -> vary = False
v_parametros.add('L0', value = L0, vary = False)
v_parametros.add('M0', value = L0, vary = False)
v_parametros.add('I0', value = L0, vary = False)
# añadimos los parámetros a ajustar (i.e. vamos a encontrar su valor)
v_parametros.add('omega_1', value = 1, min=1, max=2)
v_parametros.add('omega_2', value = 1, min=1, max=2)
v_parametros.add('omega_3', value = 1, min=1, max=2)
v_parametros.add('g', value = 1, min=1, max=2)
v_parametros.add('s', value = 1, min=1, max=2)
# ? -> q valor les voy a dar y cmo los voy a restringir.



# 4. instanciamos a minimize qn toma una fx objetivo y calcula el array a ser minimizado
    # retona -> objeto MinimizerResult

    #resultado = minimize(residuo, v_parametros, args=(t_medido, y_data), method='powell')

# obs: t_medido & y_data es la DATA q yo tgo -> q aún no la tgo definita antes.



# 5. # check results of the fit -> ocupo 'resultado' q se obtuvo ocupando clase 'minimize'

    #fitted_data = sol_ode_en_t(t_medido, y0, resultado.v_parametros)

# obs: '9.' es un float
# ? -> no toy segura sobre 'resultado.v_parametros'


# 6. Plot fitter data
    #plt.plot(t_medido, fitted_data[:, 1], '-', linewidth=2, color='red', label='fitted data')
plt.legend()


# 7 # display fitted statistics
    #report_fit(resultado) # arroja un printeable q me describe muchas cosas sobre el fit


######
plt.show()


