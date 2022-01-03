import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
from scipy.integrate import solve_ivp
from cargar_datos_pacientes import cargar_datos_pacientes
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



#########################################################
#                       CÓDIGO                          #
#########################################################

