import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
from scipy.integrate import solve_ivp
from cargar_datos_pacientes import cargar_datos_pacientes
from ODEs_y_rad_v3_Parameters import emulador_odeint
import parametros as p


#########################################################
#               AJUSTE PARÁMETROS HCC                   #
#########################################################
#                 VERSIÓN N°1 (2DA)                     #
#########################################################


#### ARCHIVOS QUE USA
# 1. importa la función cargar_datos_pacientes para leer un excel donde se
# encuentra el conteo de L de UN paciente antes-durante-dsp rad

# 2. parametros.py para las constantes

# 3. Trabaja a partir de lo que le retorna emulador_odeint del módulo
# ODEs_y_rad_v3_Parameters


#########################################################
#                       CÓDIGO                          #
#########################################################


######### SECCIÓN ############
#### AJUSTE DE PARÁMETROS ####
##############################


## TODO 04/01/2022 #1
## P. patterns of faliure (Pablo Cancer UC)
    ## Continuare trabajando en este código PARA TESTEAR
    ## fx emulador_odeint hasta que entienda como ajustar alpha_T
    ## q sospecho puede ser con el uso de los "patters of faliure"


def residuo(parametros, t, data):
    '''Calcula el residuo entre la data (datos empíricos) con los resultados
    que entrega la simulación de ODE en los t que se tiene para la data.
    El residuo lo retorna como un array, x medio de la fx ".ravel()",
    que será minimizado para encontrar el mejor fit del modelo a los datos empíricos.

    Obs sobre el flujo: La minimización del residuo, y plt el ajuste de parámetros,
    se realiza con la fx "minimize()" del módulo lmfit'''


    # todo: cdo tga la respuesta sobre el uso de los "patters of failure" #1.1
    # cambiar el nombre del parámetro "parametros" aquí, en rhs y en emulador_odeint
    # por "parametros_no_rad" para señalas que son todos los parámetros a ajustar
    # excepto alpha_T & alpha_L


    ## FLUJO:
    # Le entrego los valores iniciales como objeto Parameters fijo (fixed value)
    y0 = parametros['T0'].value, parametros['L0'].value, parametros['M0'].value, \
         parametros['I0'].value


    ## FLUJO:
    # Defino el modelo matemático de los datos (antes: f(x) = x**2 por ejemplo. ahora: ODE's)
    modelo = emulador_odeint(t, y0, parametros) # Esto arroja un arreglo (np.array)
    print(f'-----------------------------')
    print(f'modelo/resultado: {modelo}')


    ## FLUJO:
    # La data que tengo solo corresponde a los linfocitos, por lo tanto solo puedo sacar
    # el residuo entre el conteo empírico de los L y los resultados de ese conteo de L
    # que obtuve con las ODEs

    modelo_para_L = modelo[:, 1] # todo: entender q hago acá. #2  -- Listoco: explained above
        # Esto recorre cada elemento ":" del array y va extrayendo el elemento
        # en pos 1 ",1" (q es dnd están los linfocitos)


    return (modelo_para_L - data).ravel()
    # Parece calcular el residuo entre actual and fitted data



## FLUJO
# Cargar data de los linfocitos - dos arrays.
t_medido, y_medido_L = cargar_datos_pacientes("Sung figs 3 digitalized points.xlsx")
    # datos fueron obtenidos a través de la digitalización de la curva azul de la figura
    # 3b del paper HCC-Sung et al.

    # Eventualmente acá se leerán los datos de pacientes con HCC del Centro de Cancer UC
    # Data centro de cancer UC -> PENDIENTE -> Cir.L levels in blood during & after radiothe.


## FLUJO
# set parameters including bounds; you can also fix parameters using vary=False
v_parametros = Parameters()  # Guarda los parámetros y luego lo llamas cmo un dict


# Lo cual nos sirve para agregar a "v_parametros" a los parámetros con valor fijo.
# Añadimos las C.I. como parámetros fijos (FIXED)
v_parametros.add('T0', value = p.T, vary = False) # FIXED PARAMETER -> vary = False
v_parametros.add('L0', value = p.L, vary = False)
v_parametros.add('M0', value = p.M, vary = False)
v_parametros.add('I0', value = p.I, vary = False)


# Además:
# Añadimos los parámetros a ajustar EN ESTE CICLO (i.e. vamos a encontrar su valor)
v_parametros.add('omega_2', value = 1, min=10 **(-3), max=10)
v_parametros.add('omega_3', value = 1, min=10 **(-3), max=10)
v_parametros.add('g', value = 1, min=10 ** 8, max=10 ** 14)
v_parametros.add('s', value = 1, min=0, max=10 ** 14) # P. restricciones las inventé
v_parametros.add('omega_1', value = 1, min=10 **(-3), max=10) # se volverá a ajustar cn GRID SEARCH


## FLUJO
# Instanciamos a minimize qn toma una fx objetivo, la función denominada "residuo" en nuestro caso,
# qn calcula un array, que es el RESIDUO entre la data y los resultados que entrega el
# modelo ODE's + rad. Luego, minimize buscará MINIMIZAR aquel residuo (array)

    # Obs sobre minimize:
        # 1. clase 'minimize' (recordar q hay otra q se llama Minimize)
        # 2. Le tengo que pedir cn q MÉTODO ENCONTRARÁ los valores de los parámetros q busco definir
            # En nuestro caso "powell"

resultado = minimize(residuo, v_parametros, args=(t_medido, y_medido_L), method='powell')
print('im done con RESULTADO')
    # Obs:
    # t_medido & y_medido_L son los datos empíricos
        # (por ahora la curva fitteada de la fig. 3b paper HCC sung)
    # residuo tiene el modelo: ODE's + rad




########### SECCIÓN #############
### ESTADÍSTICA DE RESULTADOS ###
#################################

## display fitted statistics
report_fit(resultado) # arroja un printeable q me describe muchas cosas sobre el fit





############## SECCIÓN ###############
### CHECKEAR RESULTADOS DEL FITTEO ###
######################################

## FLUJO
# Una vez que ya encontramos los mejores parámetros resolvemos con esos valores:

 # check results of the fit -> ocupo 'resultado' q se obtuvo ocupando clase 'minimize'

y0 = np.array([p.T, p.L, p.M, p.I])
    # pero ojo q dentro de las funciones se ocupan las C.I como objetos clases Parameter
    # dentro de una tupla llamada y0 --> ? no tendré problemas con eso?

fitted_data = emulador_odeint(t_medido, y0, resultado.params)
    # obs: minimizer retorna un objeto de clase MiniizerResult
    # y para obtener los parámetros debemos llamar a su atributo "params"
    # todo
    ## AttributeError: 'MinimizerResult' object has no attribute 'v_parametros'


    # ese y0 es el array que definimos como variable global.
    # no son los objetos Parameter q ocupan las funciones antes.





######### SECCIÓN ############
### GRÁFICOS DE RESULTADOS ###
##############################


## Gráfico Data:
plt.figure() # -> ? NO SÉ K HACE ESTO
plt.scatter(t_medido, y_medido_L, marker='o', color='b', label='measured data', s= 30)
    # plt.scatter -> A scatter plot of y vs. x with varying marker size and/or color.
    # Se hace para ver en el plot la data REAL junto cn el mjr fitteo q encontramos.


## Gráfico fitted results:
# Plot fitted data
plt.plot(t_medido, fitted_data[:, 1], '-', linewidth=2, color='red', label='fitted data')
plt.legend()

## Show plots
plt.show()








