import numpy as np
from cargar_datos_pacientes import cargar_datos_pacientes


#########################################################
#                   PARÁMETROS HCC                      #
#########################################################
#                 VERSIÓN N°1 (2DA)                     #
#########################################################

i = 0

##################################################################
######## CONDICIONES INICIALES Y PARÁMETROS ODE's + RAD ##########
##################################################################

#### C.ios ################################

# T:(primary) Tumor cells
# L: Linfocitos
# M: Metastatic Tumor Cells
# I: Inactivated tumor cells realising antigens

T = 1.07 * 10 ** 11   # obs. Este valor se modifica a lo largo de código
L = 5.61 * 10 ** 9
M = 1.07 * 10 ** 8    # Patient with metastasis
I = 0                 # T deaths = 0 at the beginning


#### Parametros ################################

#En general tenemos dos tipos de parametros
# 1. Disease specific: omega_1, omega_2, omega_3, g, s, alpha_T, alpha_L
# Estos serán de la clase Parameter y buscaremos ajustar su valor.
# 2. Fixed: a, alpha_T/beta_T, f, r


######### 1. DISEASE SPECIFIC PARAMETERS

# Se definen en el código principal


######### 2. FIXED PARAMETERS OBTAINED FROM LITERATURE

#### PARÁMETROS RADIATION KILL

# T, I
    # todo
alpha_T = .037  ##[Gy**-1] #tumor-LQ(linear quadratic) cell kill
    # este valor solo se ocupa para definir al "beta_T" y definir el valor
    # del "alpha_T" que va en "v_parametros". Pero en el flujo, se ocupa este último
    # y no directamente esta variable de acá.

beta_T = alpha_T/14.3  ##[Gy**-2]

# L
#alpha_L se define en la función rad_linfo
    ## Recomendación ignacio 04/01/2022
    # dejar alpha_L con un valor un poco mayor o igual a alpha_T
        # mjr igual al ppio.



#### PARÁMETROS ODE'S

#PARÁMETROS T
a = .01  ##[days**-1] #Tumor growth


#PARÁMETROS L
f = .033 ##[days**-1]  #lymphocyte decay rate
         #OBS! acoplada a I


#PARÁMETROS M
    #Same as T


#PARÁMETROS I
r = 0.14 ##[days**-1] #half life of 5 days


##################################################################
###################   RADIACIÓN   Y   ODES   #####################
##################################################################


#### INPUTS Y VALORES INICIALES

t = 0 # día/tiempo inicial de evaluación


#### RADIATION KILL

dosis_total = 58
    # Asumimos que siempre dosis_total != 0:
fractionation = 15
dosis_pendientes = fractionation
D_T = dosis_total / fractionation #dosis homogenously delivered


#dias_irradia_week = np.array(input('Qué dias irradiar? 0=Lunes,...,6=Domingo').split(',')).astype(int)
    # se espera algo del formato '1,4,6'
dias_irradia_week = np.array([0,1,2,3,4]) #dias que se irradian en una semana

            #FUTURO: quizá poner un filtro de errores, cmo si ponen dos 4 etc.


    #Funcionamiento
        #ocupando la operación módulo (resto d la división) con el tiempo = n°dias, obtenemos
        #cmo resultdo de 0,1...6, lo q me permite saber en q día d la semana estamos y por tanto
        #'if t%7 in dias_irradia_week, entonces ese día toca irradiar.

        #comprobado 'manualmente' :)

# ITERACIONES TOTALES -- P. una función que determine el valor según largo cant datos.
iteraciones_tot = 100 # cuantas veces se resuelve la EDO! (si hay 15 dosis, entonces [15dosis+edo] + [t - iteraciones_tot edo])



#### RADIACIÓN Y ODE
sol_y = np.array([])
sol_y_T = np.array([T])
sol_y_L = np.array([L])
sol_y_M = np.array([M])
sol_y_I = np.array([I])
sol_t = np.array([0])


##################################################################
###################      DATA L MEDIDOS      #####################
##################################################################

# PASO_2. measured DATA(data to be fitted) + plot de data
    # - FORMATO: ARRAY.
    # Data centro de cancer UC -> PENDIENTE -> Cir.L levels in blood during & after radiothe.

    # temporal: datos de sung. - FORMATO: ARRAY
    # ? - para reproducir los resultados de sung necesito ocupar fig 3a o 3b ?
t_medido, y_medido_L = cargar_datos_pacientes("Sung figs 3 digitalized points.xlsx")
        # OBS: Ocuparé x el momento resultaddo fig 3b para reproducir resultados de sung.