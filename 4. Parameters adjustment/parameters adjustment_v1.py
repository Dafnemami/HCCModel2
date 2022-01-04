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


    ## FLUJO:
    # La data que tengo solo corresponde a los linfocitos, por lo tanto solo puedo sacar
    # el residuo entre el conteo empírico de los L y los resultados de ese conteo de L
    # que obtuve con las ODEs

    modelo_para_L = modelo[:, 1] # todo: entender q hago acá. #2  -- Listoco: explained above
        # Esto recorre cada elemento ":" del array y va extrayendo el elemento
        # en pos 1 ",1" (q es dnd están los linfocitos)


    return (modelo_para_L - data).ravel()
    # Parece calcular el residuo entre actual and fitted data
