#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
from scipy.integrate import solve_ivp
from cargar_datos_pacientes import cargar_datos_pacientes


#########################################################
#               AJUSTE PARÁMETROS HCC                   #
#########################################################

'''
Hay dos formas para resolver edos de esa libreria:
 (1) odeint -> más vieja
 (2) solve_ivp -> + nueva y cn más opciones. -> YO OCUPO ESTA
'''

####           FLUJO DE ESTE ARCHIVO               #####
#Definimos la función que le entrega el lado derecho de cada EDO, de un sistema
# de EDO's de primer orden, a solve_ivp.
#(Esto ya lo había hecho en el código ppal, ahora le agregaré manejo de excepciones xq
# si y le pediré q reciba los parámetros a ajustar como objetos de clase Parameters)


####          SINTAXIS          ####
#  -- ##
#  -- ?
#  -- P.


#########################################################
#                       CÓDIGO                          #
#########################################################

## PARÁMETROS OBTENIDOS DE LA LITERATURA (4, fixed):

#Tumor growth
a = .01  ##[days**-1]

#alpha_T -> tumor-LQ(linear quadratic) cell kill
alpha_T = .037  ##[Gy**-1]
    # Paper hace: Por ahora es un valor escogido basado en HCC data
                # Luego se ajusta cn GRID SEARCH
    # Obs: aún no lo ocupo xq no he incluido rad


beta_T = alpha_T/14.3  ##[Gy**-2]
f = .033 ##[days**-1]  #lymphocyte decay rate
r = 0.14 ##[days**-1] #half life of 5 days

# ? - Qué valor tomo de alpha_L antes de grid search???
# x mientras ocuparé el ya encontrado supongo.


### RESOLVER ODE's

### rhs = right hand side of the ode
def rhs(t, y, parametros): # LISTA
    """
    Input Ode: Fx entrega odes a solve_ivp en el formato pedido.
    # parámetros (a ajustar) son objetos de la clase Parameters
    """

    #print(f'y: {y}') # P. está llamando a esta fx una cant enferma de veces
    T_count, L_count, M_count, I_count = y      # variables
    "obs: C.I.'s vienen cmo un array"


    try:  # parámetros a ajustar
        omega_2 = parametros['omega_2'].value
        omega_3 = parametros['omega_3'].value
        g = parametros['g'].value
        s = parametros['s'].value

        omega_1 = parametros['omega_1'].value # se vuelve a ajustar cn GRID SEARCH

        #P. obs: alpha_T & alpha_L aparecen solo en la parte de radiación
        # alpha_T = parametros['alpha_T'] FALTA Q AGREGUE LA RADIACIÓN
        # alpha_L se busca con grid search

    except KeyError: # uso incorrecto o inválido de llaves (keys) en diccionarios
        omega_2, omega_3, g, s, omega_1 = parametros


    # ODE's -> SIN COMPONENTE RADIACIÓN
    T_dot = a * T_count - omega_1 * ( T_count/(g+T_count+M_count) ) * L_count

    L_dot = omega_2 * ( (T_count + M_count)/(g+T_count+M_count) ) * L_count \
            + omega_3 * ( I_count/(g+I_count) )* L_count + s - (f * L_count)

    M_dot = a * M_count - omega_1 * ( M_count/(g+T_count+M_count) ) * L_count

    I_dot = - r * I_count


    # OUTPUT -> no puede cambiarse.
    dydt = np.array([T_dot, L_dot, M_dot, I_dot])
    return dydt




## Solución de la EDO para UN t
def sol_ode_en_t(t, y0, parametros): # y(t)
    """
    - Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0 [para UN t.]
    - ocuparemos solve_ivp
    """

    # obs:
    # ANTES -> y0 = np.array([T,L,M,I]) #Condiciones iniciales
    # -> AHORA SE LA ENTREGAMOS A LA FX 'sol_ode_en_t' CMO ARGUMENTO

    #print(f't: {t}') # tipo  <class 'numpy.ndarray'>


    # FLUJO:
    # Recibimos un 't' que es un array de shape = (83,)
    # Creamos un array que vaya a contener todos los resultados para L con las EDO.
    sol_y_model = np.array([[y0[0], y0[1], y0[2], y0[3]]]) # agregamos C.I. al array
    # Evaluamos solve_ivp para cada tiempo
    for i in range(len(t)-1):
        # así respeto los intervalos de tiempo dados entre dos t en t_medido.
        if i != 82:
            sol = solve_ivp(rhs, (t[i], t[i+1]), y0, t_eval = np.array([t[i+1]]),
                            max_step = 0.001, args = (parametros, ))
        # solve_ivp arroja muchas cosas cmo un reporte general
            aux_sol_y_model = np.array([ sol.y[0][0], sol.y[1][0], sol.y[2][0], sol.y[3][0] ])
            sol_y_model = np.append(sol_y_model, aux_sol_y_model)
        # No entiendo xq sol_y_model queda como un array enorme sin sub arrays de 4 elementos
        # si cuando imprimo sol.y se ve bn.
        # P. VOY A SEGUIR TENIENDO EL PROBLEMA CON y_model SI NO ARREGLO ESTO PRIMERO.
        #print(f'sol.y: {sol.y}')
    #print(sol)
    print(f'sol_y_model: {sol_y_model}')

    ####### IMPORTANTE CUANDO VUELVAS A TRABAJAR EN EL CÓDIGO.
    # obs: SOLUCIÓN ALTERNATIVA.
        # Hacer derechamente el array solo de L y no hacer la tontera del y_model q me
        # está molestando.

    ########SOBRE solve_ivp #####################################
    #LO NUEVO -> 'args=(parametros,)'
    #t_eval = t, q me entrega, en q se evalua edo (puede evaluar en más puntos, xq eso lo define
    # inteligentemente solve_ivp internamente, solo q no me los entrega); ese t tiene q estar
    # dentro d (t,t+1) solve_ivp integra, i.e. obtiene sols en (t,t+1) y luego evalua eso
    # en t_eval.
    #Devuelve sol.y -> array [[T][L][M][I]]) , sol.t -> array [t]
    #############################################################

    # P. Aquí actualizar "sol" cn la influencia de la radiación.


    return sol_y_model



## Función que calcula el residuo entre la data q tgo cn la fitteada sg el método escogido
def residuo(parametros, t, data):
    """
    Compute the residual (¿cmo se obtiene este residuo?) between actual data and fitted data
    """

    # Le entrego los valores iniciales como objeto Parameters fijo (fixed value)
    y0 = parametros['T0'].value, parametros['L0'].value,\
         parametros['M0'].value, parametros['I0'].value

    # Defino el modelo matemático de los datos (antes: f(x) = x**2 por ejemplo. ahora: ODE's)
    model = sol_ode_en_t(t, y0, parametros) # Esto arroja un arreglo (np.array)

    #P. # y por una razón q aún no entiendo (esto lo extraje de "Ej minimizer & Odes.py") vamos a:
    # Recorrer cada elemento ":" del array y extraer de eso el elemento en pos 1 ",1"

    #print(f'type_model: {type(model)}')
    #print(f'model: {model}')

    # Esto recorre cada elemento ":" del array y va extrayendo el elemento en pos 1 ",1"
    y_model = model[:, 1]
    #print(f'y_model: {y_model}')
        # aRROJA:
    # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

    return (y_model - data).ravel()
    # Parece calcular el residuo entre actual and fitted data}
         # ? -> Debe ser un método de alguna clase o algo q no identifico bn, será un built in?



## LISTAS LAS FXNES. AHORA LA DATA Y PLOTTEOS


# PASO_1. Condiciones iniciales
    # T:(primary) Tumor cells
    # L: Linfocitos
    # M: Metastatic Tumor Cells
    # I: Inactivated tumor cells realising antigens

T0 = 1.07 * 10 ** 11
L0 = 5.61 * 10 ** 9
M0 = 1.07 * 10 ** 8    # Patient with metastasis
I0 = 0                 # T deaths = 0 at the beginning
y0 = np.array([T0, L0, M0, I0]) # ! - Se ocupara más abajo en el punto 5.
    # pero ojo q dentro de las funciones se ocupan las C.I como objetos clases Parameter
    # dentro de una tupla llamada y0.


# PASO_2. measured DATA(data to be fitted) + plot de data
    # - FORMATO: ARRAY.
    # Data centro de cancer UC -> PENDIENTE -> Cir.L levels in blood during & after radiothe.

    # temporal: datos de sung. - FORMATO: ARRAY
t_medido, y_medido_L = cargar_datos_pacientes("Sung figs 3 digitalized points.xlsx")
#print(type(t_medido))
#print(type(y_medido_L))

plt.figure() # -> ? NO SÉ K HACE ESTO
plt.scatter(t_medido, y_medido_L, marker='o', color='b', label='measured data', s=30)
    #plt.scatter -> A scatter plot of y vs. x with varying marker size and/or color.
    # Se hace para ver en el plot la data REAL junto cn la fitter q estamos buscando




# PASO_3. set parameters including bounds; you can also fix parameters (use vary=False)

v_parametros = Parameters() #variable parametros, así no se cambia su nombre en las fxnes.
    #Guarda los parámetros y luego lo llamas cmo un dict

# Añadimos las C.I. como parámetros fijos (FIXED)
v_parametros.add('T0', value = T0, vary = False) # FIXED PARAMETER -> vary = False
v_parametros.add('L0', value = L0, vary = False)
v_parametros.add('M0', value = M0, vary = False)
v_parametros.add('I0', value = I0, vary = False)
# Añadimos los parámetros a ajustar EN ESTE CICLO (i.e. vamos a encontrar su valor)
v_parametros.add('omega_2', value = 1, min=10 **(-3), max=10)
v_parametros.add('omega_3', value = 1, min=10 **(-3), max=10)
v_parametros.add('g', value = 1, min=10 ** 8, max=10 ** 14)
v_parametros.add('s', value = 1, min=0, max=10 ** 14) # P. constraits los inventé
v_parametros.add('omega_1', value = 1, min=10 **(-3), max=10) # luego se ajusta cn GRID SEARCH




# PASO_4. instanciamos a minimize qn toma una fx objetivo y calcula el array a ser minimizado
    # retona -> objeto MinimizerResult

resultado = minimize(residuo, v_parametros, args=(t_medido, y_medido_L), method='powell')

    # t_medido & y_medido_L es la DATA q yo tgo (por ahora lo del paper)
    # residuo tiene el modelo : edo's + P. radiación

'''
A la clase 'minimize' (recordar q hay otra q se llama Minimize)
le entrego el residuo a minimizar, los parámetros a definir y la data q tgo. 

Así tmbn le tengo que pedir cn q MÉTODO ENCONTRARÁ los valores de los parámetros q busco definir
'''


# PASO_5. # check results of the fit -> ocupo 'resultado' q se obtuvo ocupando clase 'minimize'
fitted_data = sol_ode_en_t(t_medido, y0, resultado.v_parametros)
    # ese y0 es el array que definimos como variable global.
    # no son los objetos Parameter q ocupan las funciones antes.

# obs: '9.' es un float




# PASO_6. Plot fitted data
plt.plot(t_medido, fitted_data[:, 1], '-', linewidth=2, color='red', label='fitted data')
plt.legend()



# PASO_7 # display fitted statistics
report_fit(resultado) # arroja un printeable q me describe muchas cosas sobre el fit


######
plt.show()


