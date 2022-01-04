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

## TODO 04/01/2022
## P. patterns of faliure (Pablo Cancer UC)
    ## Continuare trabajando en este código PARA TESTEAR
    ## fx emulador_odeint hasta que entienda como ajustar alpha_T
    ## q sospecho puede ser con el uso de los "patters of faliure"