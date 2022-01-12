import numpy as np
from scipy.integrate import solve_ivp #solve_ivp tiene más cosas q odeint

from Pandas_y_linfocitos_Parameters import rad_linfo
import parametros as p
from crear_array_t_eval import crear_array_t_eval


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

        # P. alpha_T es un valor dado y no lo voy a ajustar con este método por ahora

            ########################################
            # ? - No sé como ajusta con el método de powell para la minimización
            # del residuo a parámetros que se encuentran fuera de las ODEs
            # acá señaladas. Q es lo q ocurre con alpha_T

            # Puedo trabajar con valores iniciales alpha_T=alpha_L = .037
            # pero dsp de eso no sé q hacer para fittear alpha_T

            #  Quizá los "patters of faliure" me puedan ser útiles para ajustar alpha_T
            ########################################

        # alpha_T se busca con grid search
        # alpha_L se ocupa solo en la rad

    except KeyError: # uso incorrecto o inválido de llaves (keys) en diccionarios
        omega_2, omega_3, g, s, omega_1 = parametros


    # ODE's -> S/RADIACIÓN
    T_dot = p.a * T_count - omega_1 * ( T_count/(g+T_count+M_count) ) * L_count

    L_dot = omega_2 * ( (T_count + M_count)/(g+T_count+M_count) ) * L_count \
            + omega_3 * ( I_count/(g+I_count) )* L_count + s - (p.f * L_count)

    M_dot = p.a * M_count - omega_1 * ( M_count/(g+T_count+M_count) ) * L_count

    I_dot = - p.r * I_count


    # OUTPUT -> no puede cambiarse.
    dydt = np.array([T_dot, L_dot, M_dot, I_dot])

    return dydt



def emulador_odeint(t: np.array, y0, parametros): # y(t)
    # antes "sol_ode_en_t" en 'ODEs_y_rad_v2.py'

    ''' Esto es necesario porque la función residuo necesita una función que
     represente el modelo. ej de un modelo f(x) == x**2
    - Solution to the ODE y'(t) = f(t,y,k) with initial condition y(0) = y0
    para un array de tiempos a evaluar uno a uno con solve_ivp.

    1. t --> <class 'numpy.ndarray'>

    2. y0 = np.array([T,L,M,I]) #Condiciones iniciales
          --> este parámetro NO SE OCUPA, pues el y0 q me sirve es el de dsp de la primera
          dosis de rad, y el que le llega a esta fx corresponde a antes de ello.
          --> igual me serviria para definir "#### C.ios ########" pero decidí
          obtener esos valores de "parametros.py"

    3. parametros --> los recibe para que solve_ivp se los entregue a rhs

    :return: emular el array q retornaria odeint cuando t_array == t_medido.
      i.e. un array de arrays del tipo [T, L, M, I] de largo t_data (t_medido)
    '''

    #print(f'largo t_medido: {len(t)}')
    #print(f't_medido: {t}')

    #### C.ios ########
    T = y0[0] ## ANTES: importaba datos desde "parametros.py"
    L = y0[1]
    M = y0[2]
    I = y0[3]
    dia_actual = p.t # día/tiempo inicial == 0

    #### Arrays a retornar
    sol_y = p.sol_y  # solo este me interesa retornar.
    sol_y_T = p.sol_y_T
    sol_y_L = p.sol_y_L
    sol_y_M = p.sol_y_M
    sol_y_I = p.sol_y_I
    sol_t = p.sol_t

    #### Tiempos a evaluar
    'llega t_medido==datos empíricos (un array), pues esta función antes contenía a ' \
    'odeint el cual trabaja con t==array.'


     #### Flujo del siguiente bloque de código:
        # Se irradia a las 00hrs del día 1 - cambia T, L, I
        # Se simula la ODE entre las 00hrs y las 23:59 del dia 1 - cambia T, L, I, M
        # ODE es **evaluada** en todos los t que pertenezcan a las 00hrs y las 23:59 del dia 1
            # evaluada == soluciones q retorna
            # obj: ODE se evalua en los t que la data recibida del paciente
                # P/? - operación módulo seguirá funcionando correctamente?


    #largo_t_aux = 0


    # si el array del tiempo viene desordenado:
    t.sort()


    # parámetros a ajustar/ocupar en la rad
    try:
        alpha_L = parametros['alpha_L'].value
        alpha_T = parametros['alpha_T'].value
        print(f'alphas _T: {alpha_T} ; _L: {alpha_L}')
    except KeyError: # uso incorrecto o inválido de llaves (keys) en diccionarios
        omega_2, omega_3, g, s, omega_1, alpha_L, alpha_T = parametros


    for iteraciones in range(int(t[-1]) + 1):
        #print(f'iteraciones {iteraciones}')
        # Deben haber tantos ciclos de este for como días deban pasar.
        # tomo el último n° del array y le saco la parte entera.
            # obs: dias == cant de veces q se aplica rad =! veces q se resuelven ODE's

        ###############################################################################
        ## FLUJO "for + range(+1)"

        # dia_actual == 0, iteraciones valor inicial == 0
        # Comienza buscando tiempos en que evaluar las edo que correspondan al
        # dia 0, i.e. tiempos en el array "t_medido" que llega que empiecen con cero

        # Al final, para que efectivamente evalue lo que ocurre el último día,
        # 293 por ejemplo, es necesario que "range" del "for" anterior lleve un "+1"
        # de lo contrario no evalua en las odes los tiempos asociados al dia 293.

        # En otras palabras, evaluamos t[-1] + 1 días, pues el dia 0,
        # es realmente el primer día
        ###############################################################################

        # SECCIÓN: RADIATION KILL'S RESOLUTION PER TYPE OF CELL

        if p.dosis_total != 0: # si hay dosis se irradia, sino solo se resuelve la edo directamente.

            if p.dosis_pendientes != 0 and dia_actual%7 in p.dias_irradia_week:
                    # "dia_actual%7": evalua si dia t toca irradiar o no

                ## FLUJO
                # dia_actual siempre serán números enteros, plt la operación módulo funcionará como
                # se espera. Ahora los tiempos en que evaluaremos las ODE no necesariamente serán
                # enteros, por lo q deberemos procurar obtener todas esas soluciones de las ODE,
                # i.e. todas las soluciones de las ODE durante un día, antes de pasar al siguiente
                # dia y volver a aplicar la radiación a las 00hrs.

                    ## ¡¡ IMPORTANTE !!
                    # Lo anterior me debería asegurar que se aplica radiación
                    # solo UNA vez y a las 00hrs

                p.dosis_pendientes-=1 # Se van descontando bn ;)

                # I produced by rkill
                I = I + T * (1 - np.exp(- alpha_T * p.D_T - p.beta_T * p.D_T ** 2))
                # T alive after rkill
                T = T * np.exp(- alpha_T * p.D_T - p.beta_T * p.D_T ** 2)

                # L alive after rkill
                L = rad_linfo('DVH acumulado - pares x,y.xlsx', [0,1], L, alpha_L)
                    #TOMANDO Linicial como L dsp de una iteración EDO (con s=0), se comprobó
                    # que este valor es correcto dsp de la primera radiación.

                # Obs: En el caso de RAD, se actualizan las nuevas C.I. al resolver
                # las ecs. de rad.


        # SECCIÓN: Ode's

        # Creamos un array con todos los tiempos que están dentro de dia_actual
            # i.e definimos un t_eval para solve_ivp personalizado para c/día.

        v_array_t_eval = crear_array_t_eval(dia_actual, t)
        #print(f'iteración n°: {iteraciones}')
        #print(f'v_array_eval: {v_array_t_eval}')

        #largo_t_aux += len(v_array_t_eval)
        #print(f'largo_t_aux: {largo_t_aux}')

        ## Sobre el por qué del siguiente FLUJO:
        # Obs: Hay veces que ocurrirá que no hay datos empíricos sobre un día en pclar
        # En estos casos debemos aplicar radiación al paciente más no resolver las ODE's
        # ni guardar aquellos resultados en los arrays


        if len(v_array_t_eval) != 0:

            for t_a_evaluar in v_array_t_eval:
                #print(f'im in "for t_evaluar"')


                y0 = np.array([T, L, M, I])  # Condiciones iniciales
                # La fx recibe y0, pero no me sirven xq son antes de pasar por la rad.

                sol = solve_ivp(rhs, (dia_actual, dia_actual+1), y0,
                                t_eval = np.array([t_a_evaluar]), max_step = 0.001, args=(parametros, )
                                )

                    # obs: en solve_ivp "arg" son entregados a la fx "rhs"

                #t_eval = t, q me entrega, en q se evalua edo (puede evaluar en más puntos, xq eso lo define
                # inteligentemente solve_ivp internamente, solo q no me los entrega); ese t tiene q estar
                # dentro d (t,t+1) solve_ivp integra, i.e. obtiene sols en (t,t+1) y luego evalua eso
                # en t_eval.
                #Devuelve sol.y -> array [[T][L][M][I]]) , sol.t -> array [t]

                #Añadir resultados a array para gráficar
                    # solo este me interesa retornar.
                     # (solve_ivp arroja muchas cosas cmo un reporte general "sol.y")


                #Funciona para data (arrays del t) de !=s largos :) 100% comprobado
                if len(sol_y) == 0:
                    # obs:
                    # Antes este if era "iteraciones==0 and len(sol_y) == 0"
                    # pero eso provocaba que durante la iteración cero, donde dia_actual ==0
                    # si había más de un tiempo que evaluar, solo se guardara el último
                    # pues siempre entraba a este "if" y no al "elif" de acá abajo
                    sol_y = np.array([ sol.y[0][0], sol.y[1][0], sol.y[2][0], sol.y[3][0] ])
                elif len(sol_y) != 0:
                    sol_y = np.vstack( (sol_y,
                                        np.array([sol.y[0][0], sol.y[1][0],
                                                  sol.y[2][0], sol.y[3][0]])) )
                        # np.vstack concatena arrays en vertical sin juntarnos en un solo arrays,
                        # q es lo q hace "append"


                #print(f'sol_y after ite n°{iteraciones}: {sol_y}')


                sol_y_T = np.append(sol_y_T, sol.y[0]) # Append para arrays
                sol_y_L = np.append(sol_y_L, sol.y[1]) #(lo q tgo, lo q quiero agregar)
                sol_y_M = np.append(sol_y_M, sol.y[2])
                sol_y_I = np.append(sol_y_I, sol.y[3])
                sol_t = np.append(sol_t, sol.t)  #sol.t guarda los t q le doy a t_eval en solve_ivp


                # T,L,M,I: Actualizar CI para sgte iteración
                T, = sol.y[0]    #T,L = (T,L) 'Tupla'; Si [a,b] => T,L = [a,b] es T = a y L = b
                L, = sol.y[1]
                M, = sol.y[2]
                I = sol.y[3][0]  #Dos formas distintas de extraer el n° del array d 1d que devuelve.
                                # i.e. xq una tupla de un elemento necesita la coma tipo A,

                # Obs: En el caso de RAD, se actualizan las nuevas C.I. al resolver
                # las ecs. de rad.

        #print(f'iteración {iteraciones}')
        #print(f'sol_y {sol_y}')

        dia_actual += 1 # para que en el siguiente intervalo se evalue en el día siguente

    #print(f'len(sol_y): {len(sol_y)}')

    return sol_y




if __name__ == "__main__":
    # Queremos asegurarnos que fx emulador_odeint retorna lo mismo que odeint:

    pass
