import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp #solve_ivp tiene más cosas q odeint

from Pandas_y_linfocitos import rad_linfo
from lmfit import Parameters
import parametros as p


## Repetiremos lo del archivos "ODEs_y_rad_v2.py" pero usando la clase Parameters
## para que los resultados puedan ser usados para el ajuste de parámetros.
## El objetivo es crear una función que retorne lo q la función residuo necesita


# Primer paso -- que reproduzca los resultados de "ODEs_y_rad_v2.py"
# 2do paso -- cambiar los valores de las constantes x los a utilizar para
            # el ajuste de parámetros.


#### RESOLVER ODE's

def rhs(t, y, parametros): # right hand side of the ode
    """
    entrega ode's a solve_ivp en el formato pedido.
    # parámetros (a ajustar) = objetos de la clase Parameters
    """

    T_count, L_count, M_count, I_count = y      # variables

    try:  # parámetros a ajustar
        omega_2 = parametros['omega_2'].value
        omega_3 = parametros['omega_3'].value
        g = parametros['g'].value
        s = parametros['s'].value
        omega_1 = parametros['omega_1'].value # se vuelve a ajustar cn GRID SEARCH

        alpha_T = parametros['alpha_T'] # RADIACIÓN
        # alpha_L se busca con grid search

    except KeyError: # uso incorrecto o inválido de llaves (keys) en diccionarios
        omega_2, omega_3, g, s, omega_1, alpha_T = parametros


    # ODE's -> S/RADIACIÓN
    T_dot = p.a * T_count - omega_1 * ( T_count/(g+T_count+M_count) ) * L_count

    L_dot = omega_2 * ( (T_count + M_count)/(g+T_count+M_count) ) * L_count \
            + omega_3 * ( I_count/(g+I_count) )* L_count + s - (p.f * L_count)

    M_dot = p.a * M_count - omega_1 * ( M_count/(g+T_count+M_count) ) * L_count

    I_dot = - p.r * I_count


    # OUTPUT -> no puede cambiarse.
    dydt = np.array([T_dot, L_dot, M_dot, I_dot])

    return dydt



def emulador_odeint(t, y0, parametros): # y(t)
    # antes "sol_ode_en_t"
    ''' Esto es necesario porque la función residuo necesita una función que
     represente el modelo. ej de un modelo f(x) == x**2
    - Solution to the ODE y'(t) = f(t,y,k) with initial condition y(0) = y0
    para un array de tiempos a evaluar uno a uno con solve_ivp.

    t --> <class 'numpy.ndarray'>  --
        P. debo evaluar a la edo en los mismos tiempos q los datos,
        así q debo ocupar ese array t_medido.
    y0 = np.array([T,L,M,I]) #Condiciones iniciales

    :return: emular el array q entregaría odeint, i.e.
    un array de arrays del tipo [T, L, M, I] de largo t_data (t_medido)
    '''

    #### C.ios ########
    T = p.T
    L = p.L
    M = p.M
    I = p.I

    #### Arrays a retornar
    sol_y = p.sol_y  # solo este me interesa retornar.
    sol_y_T = p.sol_y_T
    sol_y_L = p.sol_y_L
    sol_y_M = p.sol_y_M
    sol_y_I = p.sol_y_I
    sol_t = p.sol_t

    #### Tiempos a evaluar
    'llega t_medido==datos empíricos (un array), pues esta función antes contenía a ' \
    'odeint el cual trabaja con t==array. Plt, en el codigo a continuación ' \
    'P. usaré iteradores para recorrer el array del tiempo'

    'P. !!! creo q puedo pedirle a evalue más cosas a solve_ivp dentro de un día, con t_eval,' \
    'así respetaamos aplicar la radiación cuando corresponda (i.e una vez al día en los dias q ' \
    'corresponde)'

     #### Flujo del siguiente bloque de código:
        # Se irradia a las 00hrs del día 1 - cambia T, L, I
        # Se simula la ODE entre las 00hrs y las 23:59 del dia 1 - cambia T, L, I, M
        # ODE es **evaluada** en todos los t que pertenezcan a las 00hrs y las 23:59 del dia 1
            # evaluada == soluciones q retorna

    for iteraciones in range(p.iteraciones_tot):

        # SECCIÓN: RADIATION KILL'S RESOLUTION PER TYPE OF CELL

        if p.dosis_total != 0: # si hay dosis se irradia, sino solo se resuelve la edo directamente.

            if p.dosis_pendientes != 0 and (t)%7 in p.dias_irradia_week: #"t%7": evalua si dia t toca irradiar o no

                p.dosis_pendientes-=1

                # I produced by rkill
                I = I + T * (1 - np.exp(- p.alpha_T * p.D_T - p.beta_T * p.D_T ** 2))
                # T alive after rkill
                T = T * np.exp(- p.alpha_T * p.D_T - p.beta_T * p.D_T ** 2)

                # L alive after rkill
                L = rad_linfo('DVH acumulado - pares x,y.xlsx', [0,1], L)
                    #TOMANDO Linicial como L dsp de una iteración EDO (con s=0), se comprobó
                    # que este valor es correcto dsp de la primera radiación.


        # SECCIÓN: Ode's

        y0 = np.array([T, L, M, I]) #Condiciones iniciales
            # La fx recibe y0, pero no me sirven xq son antes de pasar por la rad.

        sol = solve_ivp(rhs, (t,t+1), y0, t_eval = np.array([t+1]), max_step = 0.001)
        #t_eval = t, q me entrega, en q se evalua edo (puede evaluar en más puntos, xq eso lo define
        # inteligentemente solve_ivp internamente, solo q no me los entrega); ese t tiene q estar
        # dentro d (t,t+1) solve_ivp integra, i.e. obtiene sols en (t,t+1) y luego evalua eso
        # en t_eval.
        #Devuelve sol.y -> array [[T][L][M][I]]) , sol.t -> array [t]

        #Añadir resultados a array para gráficar
            # solo este me interesa retornar.
        sol_y = np.append(sol_y, np.array(sol.y[0][0], sol.y[1][0],
                                          sol.y[2][0], sol.y[3][0]))
            # ? - tgo la duda si va a mantener la separación [[T,L,M,I] [T,L,M,I]...]
            # o si va a dejarlo cmo [T,L,M,I,T,L,M,I]
        sol_y_T = np.append(sol_y_T, sol.y[0]) # Append para arrays
        sol_y_L = np.append(sol_y_L, sol.y[1]) #(lo q tgo, lo q quiero agregar)
        sol_y_M = np.append(sol_y_M, sol.y[2])
        sol_y_I = np.append(sol_y_I, sol.y[3])
        sol_t = np.append(sol_t, sol.t)  #sol.t guarda el t q le doy a t_eval en solve_ivp


        # T,L,M,I: Actualizar CI para sgte iteración
        T, = sol.y[0]     #T,L = (T,L) 'Tupla'; Si [a,b] => T,L = [a,b] es T = a y L = b
        L, = sol.y[1]
        M, = sol.y[2]
        I = sol.y[3][0]   #Dos formas distintas de extraer el número del array d 1d que devuelve.
                        # i.e. xq una tupla de un elemento necesita la coma tipo A,

        t += 1 #para que en el siguiente intervalor se evalue en el segundo siguiente


    return sol_y




if __name__ == "__main__":
    # Para probar si el código retorna lo q quiero.
    pass
