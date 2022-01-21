import numpy as np
import prueba_parametros as p
from prueba_Pandas_y_linfocitos_Parameters import rad_linfo
from prueba_crear_array_t_eval import crear_array_t_eval
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
from prueba_cargar_datos_pacientes import cargar_datos_pacientes


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
    print(f'T_inicial: {T}')
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
    aux_sol_y_L = p.sol_y_L
    aux_sol_t = p.sol_t

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

        # SECCIÓN: RADIATION KILLS RESOLUTION PER TYPE OF CELL

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
                    # solo UNA vez y a las 00hrs por día.

                #print(f'dosis P. : {p.dosis_pendientes}')
                p.dosis_pendientes-=1 # Se van descontando bn ;)

                # I produced by rkill
                I = I + T * (1 - np.exp(- alpha_T * p.D_T - p.beta_T * p.D_T ** 2))
                # T alive after rkill
                T = T * np.exp(- alpha_T * p.D_T - p.beta_T * p.D_T ** 2)
                #print(f'T_dsp rad: {T}')

                # L alive after rkill
                L = rad_linfo('DVH acumulado - pares x,y.xlsx', [0,1], L, alpha_L)
                    #TOMANDO Linicial como L dsp de una iteración EDO (con s=0), se comprobó
                    # que este valor es correcto dsp de la primera radiación.

                # Obs: En el caso de RAD, se actualizan las nuevas C.I. al resolver
                # las ecs. de rad.


        # SECCIÓN: ODE's

        # Creamos un array con todos los tiempos que están dentro de dia_actual
            # i.e definimos un t_eval para solve_ivp personalizado para c/día.

        v_array_t_eval = crear_array_t_eval(dia_actual, t)
        #print(f'iteración n°: {iteraciones}')
        #print(f'v_array_eval: {v_array_t_eval}')

        #largo_t_aux += len(v_array_t_eval)
        #print(f'largo_t_aux: {largo_t_aux}')

        ## Sobre el por qué del siguiente FLUJO:
        # Obs: Hay veces que ocurrirá que no hay datos empíricos sobre un día en particular
        # En estos casos debemos aplicar radiación al paciente más no resolver las ODE's
        # ni guardar aquellos resultados en los arrays.


        if len(v_array_t_eval) != 0: # Existen datos empíricos

            # Simulamos las ODE's durante un día completo

            for t_a_evaluar in v_array_t_eval:

                if t_a_evaluar == 0:
                    ## ¡¡ IMPORTANTE !!
                    # En este caso, si bien los valores actuales para TLMI corresponden a los
                    # posteriores a la primera dosis de rad en el día 0 a las 00hrs,
                    # estos datos se tendrán que comparar con la data, la cual toma como primer
                    # valos TLIM iniciales, plt guardaré esos valores cmo primer item en el
                    # array del output en lugar de los resultados de la rad.

                    sol_y = np.array([p.T, p.L, p.M, p.I]) # CI
                    #print(f't_a_evaluar: {t_a_evaluar}')
                    #print(f'T_dsp de rad: {T}')


                elif t_a_evaluar != 0:
                    #print(f't_a_evaluar: {t_a_evaluar}')
                    #print(f'T: {T}, L: {L}, M: {M}, I: {I}')
                    #print(f'T+I {T+I}')

                    y0 = np.array([T, L, M, I])  # resultados de la radiación

                    sol = solve_ivp(rhs, (dia_actual, dia_actual+1), y0,
                                    t_eval = np.array([t_a_evaluar]), max_step = 0.001,
                                    args=(parametros, ) )

                    # todo: simulación en t pertenecientes a enteros + posterior interpolación
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

                        ## ¡¡ IMPORTANTE !!
                        # Si los datos tienen un t==0, el flujo jamás debería pasar por
                        # este "if"

                        # OBS:
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
                    # todo: esto se actualiza al final de terminar todas las simulacionesde UN día!!!
                    # antes estaba actualizad estos valores c/vez q resolvía un t intermedio a un día
                    # lo q me cambiaba los resultados pues la simulación dentro del MISMO INTERVALO
                    # me da cosas distintas pues parte de valores iniciales q no le corresponden


        ## FLUJO ---
        # SALIMOS DEL CICLO "FOR + len(v_array_t_eval) != 0" PUES YA ENCONTRAMOS TODOS
        # LOS RESULTADOS DEL SIST DE ODES + RAD CORRESPONDIENTES A UN DÍA O SIMPLEMENTE
        # ES UN DÍA EN Q NO TENEMOS DATA EMÍPÍRICA Y SIMPLEMENT QUEREMOS ACTUALIZAR
        # LAS CI PARA EL SGTE DÍA

        # Actualizo los valores de TLIM como resultado de la simulación de las ODE's
        # a las 23_59hrs q serán las CI para el día sgte:

        y0 = np.array([T, L, M, I])  # resultados de la radiación o ODEs final del día anterior

        sol = solve_ivp(rhs, (dia_actual, dia_actual + 1), y0,
                        t_eval=np.array([dia_actual + 1]), max_step=0.001,
                        args=(parametros,))


        aux_sol_y_L = np.append(aux_sol_y_L, sol.y[1])
        aux_sol_t = np.append(aux_sol_t, sol.t)

        T, = sol.y[0]    #T,L = (T,L) 'Tupla'; Si [a,b] => T,L = [a,b] es T = a y L = b
        L, = sol.y[1]
        M, = sol.y[2]
        I = sol.y[3][0]  #Dos formas distintas de extraer el n° del array d 1d que devuelve.
        # i.e. xq una tupla de un elemento necesita la coma tipo A,

        # Obs: En el caso de RAD, se actualizan las nuevas C.I. al resolver
        # las ecs. de rad.


        #print(f'iteración {iteraciones}')

        dia_actual += 1 # para que en el siguiente intervalo se evalue en el día siguente


    #print(f'len(sol_y): {len(sol_y)}')

    return sol_y, aux_sol_y_L, aux_sol_t





if __name__ == "__main__":
    # Queremos asegurarnos que fx emulador_odeint retorna lo mismo que odeint:

    y0 = np.array([p.T, p.L, p.M, p.I])
    # pero ojo q dentro de las funciones se ocupan las C.I como objetos clases Parameter
    # dentro de una tupla llamada y0 --> ? no tendré problemas con eso?

    v_parametros = Parameters()  # Guarda los parámetros y luego lo llamas cmo un dict

    # Lo cual nos sirve para agregar a "v_parametros" a los parámetros con valor fijo.
    # Añadimos las C.I. como parámetros fijos (FIXED)
    v_parametros.add('T0', value=p.T, vary=False)  # FIXED PARAMETER -> vary = False
    v_parametros.add('L0', value=p.L, vary=False)
    v_parametros.add('M0', value=p.M, vary=False)
    v_parametros.add('I0', value=p.I, vary=False)

    # Además:
    # Añadimos los parámetros a ajustar EN ESTE CICLO (i.e. vamos a encontrar su valor)
    # Todas las restricciones se obtuvieron de "Table 1" paper HCC Sung

    #v_parametros.add('omega_2', value= 0.00144515 ,min=10 **(-3), max=1)
    #v_parametros.add('omega_3', value=0.009, min=10 **(-8), max=1)
    #v_parametros.add('g', value=6.3359 * 10 ** 12, min=10 ** 3, max=10 ** 14)
    #v_parametros.add('s', value=76375156.2, min=0, max=5.61 * 10 ** 19)
    #v_parametros.add('omega_1', value=0.11627178, min=10 **(-8), max=1)  # se volverá a ajustar cn GRID SEARCH
    #v_parametros.add('alpha_L', value=0.737,  min=10 ** (-2), max=1)

    #v_parametros.add('omega_2', value= 0.003 ,min=10 **(-3), max=1)
    #v_parametros.add('omega_3', value=0.009, min=10 **(-8), max=1)
    #v_parametros.add('g', value= 7.33 * 10 ** 10, min=10 ** 3, max=10 ** 14)
    #v_parametros.add('s', value=1.47 * 10 ** 8, min=0, max=5.61 * 10 ** 19)
    #v_parametros.add('omega_1', value=0.119, min=10 **(-8), max=1)  # se volverá a ajustar cn GRID SEARCH
    #v_parametros.add('alpha_L', value=0.737,  min=10 ** (-2), max=1)

    v_parametros.add('omega_2', value= 0.003, vary= False)
    v_parametros.add('omega_3', value=0.009, vary= False)
    v_parametros.add('g', value= 7.33 * 10 ** 10, vary= False)
    v_parametros.add('s', value=1.47 * 10 ** 8, vary= False)
    v_parametros.add('omega_1', value=0.119, vary= False)  # se volverá a ajustar cn GRID SEARCH
    v_parametros.add('alpha_L', value=0.737, vary= False)

    # todo: revisar xq no me da la curva q debería si ocupo los valores bn ajustados
    #

    # Añadimos alpha_T como un valor FIXED
    v_parametros.add('alpha_T', value=0.139, vary=False)  # se ajustará cn GRID SEARCH
        # todo: resolver con aplha_T = 0.139



 #########################################
    ## FLUJO
    # Cargar data de los linfocitos - dos arrays.
    t_medido, y_medido_L = cargar_datos_pacientes("Sung figs 3 digitalized points.xlsx")
    # datos fueron obtenidos a través de la digitalización de la curva azul de la figura
    # 3b del paper HCC-Sung et al.

    # Eventualmente acá se leerán los datos de pacientes con HCC del Centro de Cancer UC
    # Data centro de cancer UC -> PENDIENTE -> Cir.L levels in blood during & after radiothe.

    ## FRIENDLY REMINDER:
    # data está NORMALIZADA, plt ocuparemos:
    #y_medido_L_no_normalizado = y_medido_L * p.L
    #plt.figure()

    ## GRÁFICO DATA
    plt.scatter(t_medido, y_medido_L, marker='o', color='b', label='measured data', s=30)


    ## Gráfico fitted results:

    # t medido ideal:
    #t_medido = np.linspace(0, 200, 201)

    # solo resultados para L:
    fitted_data, aux_sol_y_L, aux_sol_t = emulador_odeint(t_medido, y0, v_parametros)
    print(f't medido: {t_medido}')

    print(f'L_aux: {aux_sol_y_L}')

    print(f'sol_t_aux: {aux_sol_t}')

    plt.plot(t_medido, fitted_data[:, 1]/p.L, '-', linewidth=2, color='red', label='fitted data')
    plt.plot(aux_sol_t, aux_sol_y_L/p.L, '-', linewidth=2, color='orange', label='aux_sol_y_L, ')

    ####################

    plt.legend()

    plt.title('prueba emulador_odeint')

    plt.grid()

    ## Show plots
    plt.show()