import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#ocupo solve_ivp en lugar de odeint xq tiene más cosas
from Pandas_y_linfocitos import rad_linfo



# PENDIENTES/PROBLEMAS CN MAYÚSCULAS.

##################################################################
######## CONDICIONES INICIALES Y PARÁMETROS RADIACIÓN  ###########
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




#### PARÁMETROS RADIATION KILL ###############

# T, I
alpha_T = .139  ##[Gy**-1] #tumor-LQ(linear quadratic) cell kill
beta_T = alpha_T/14.3  ##[Gy**-2]


# L
#alpha_L se define en la función rad_linfo





#### PARÁMETROS ODE'S $#######################

#PARÁMETROS T
a = .01  ##[days**-1] #Tumor growth
omega_1 = .119  ##[days**-1] #Tumor-directed lymphocyte efficiency
g = (7.33 * 10** 10)  ##adimensional #Half saturation constant


#PARÁMETROS L
omega_2 = .003  ##[days**-1]  #tumor-lymphocyte recruitment constant
omega_3 = .009 ##[days**-1]  #Inactivated tumor-lymphocyte recruitment constant
s = 1.47 * 10 ** 8 ##[days**-1] #lymphocyte regeneration
f = .033 ##[days**-1]  #lymphocyte decay rate
         #OBS! acoplada a I


#PARÁMETROS M
    #Same as T


#PARÁMETROS I
r = 0.14 ##[days**-1] #half life of 5 days




##################################################################
###################   RADIACIÓN   Y   ODES   #####################
##################################################################


#### INPUTS

t = 0 #tiempo inicial de evaluación



# RADIATION KILL

#dosis_total = int(input('¿Cuál es la dosis total en [Gy]? '))
dosis_total = 58

if dosis_total != 0: #si dosis_total == 0 entonces solo se resuelve la edo por un t a determinar.

    #fractionation = int(input('¿En cuántas fracciones se entrega? '))
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


# ITERACIONES TOTALES
iteraciones_tot = 100 # cuantas veces se resuelve la EDO! (si hay 15 dosis, entonces [15dosis+edo] + [t - iteraciones_tot edo])



# Input Ode: Fx entrega odes a solve_ivp en el formato pedido.

def rhs(t, y):   # rhs = right hand side of the ode

    T_count, L_count, M_count, I_count = y   #variables

    # ODE's
    T_dot = a * T_count - omega_1 * ( T_count/(g+T_count+M_count) ) * L_count

    L_dot = omega_2 * ( (T_count + M_count)/(g+T_count+M_count) ) * L_count \
            + omega_3 * ( I_count/(g+I_count) )* L_count + s - (f * L_count)

    M_dot = a * M_count - omega_1 * ( M_count/(g+T_count+M_count) ) * L_count

    I_dot = - r * I_count


    dydt = np.array([T_dot, L_dot, M_dot, I_dot])

    return dydt



#### COMENTARIO SOBRE FUNCIONAMIENTO EN GENERAL

# Si quedan fracciones de dosis que aplicar, se tiene que iterar
# el trabajo entre radiación y resolución odes.

    # Esto requiere por cada irradiación

        # ODE: cambiar condiciones iniciales
        # ODE: se resuelva UNA vez entre irradiaciones.
        # rKILL: actualizar valores de T e I.


# Si NO queda irradiación pendiente, nos saltamos el paso de
# radiación y pasamos directo a resolver las edos.



#### RADIACIÓN Y ODE

sol_y_T = np.array([T])
sol_y_L = np.array([L])
sol_y_M = np.array([M])
sol_y_I = np.array([I])
sol_t = np.array([0])
L_sabado = np.array([])######################



for iteraciones in range(iteraciones_tot):

    # SECCIÓN: RADIATION KILL'S RESOLUTION PER TYPE OF CELL

    if dosis_total != 0: # si hay dosis se irradia, sino solo se resuelve la edo directamente.

        if dosis_pendientes != 0 and (t)%7 in dias_irradia_week: #"t%7": evalua si dia t toca irradiar o no

            dosis_pendientes-=1

            # I produced by rkill
            I = I + T * (1 - np.exp(- alpha_T * D_T - beta_T * D_T ** 2))
            # T alive after rkill
            T = T * np.exp(- alpha_T * D_T - beta_T * D_T ** 2)

            # L alive after rkill
            L = rad_linfo('DVH acumulado - pares x,y.xlsx', [0,1], L)
                #TOMANDO Linicial como L dsp de una iteración EDO (con s=0), se comprobó
                # que este valor es correcto dsp de la primera radiación.


#####################   PRIMEROS 3 SÁBADOS (RESULTADO FINAL RAD)   ########################
    if t==5 or t==12 or t==19: #t+1 de la última rad
        #print(t)
        L_sabado = np.append(L_sabado, L)
###########################################################################################


            #sol_y_T = np.append(sol_y_T, T)  #append para arrays np.append(lo q tgo, lo q quiero agregar)
            #sol_y_L = np.append(sol_y_L, L)
            #sol_y_I = np.append(sol_y_I, I)
            #sol_t = np.append(sol_t, t)  # t+1 para q esté en mismo t q resultado edo d la presente iteración.


            #Para solucionar problema de dimensión cn array d M:
            #cells cn radiación tendrán más elementos en el array si no hago esto.
            #sol_y_M = np.append(sol_y_M, sol.y[2]) #ojo q aquí estaré guardandos M's iguales a tiempos distintos.


    # SECCIÓN: Ode's

    y0 = np.array([T,L,M,I]) #Condiciones iniciales
    sol = solve_ivp(rhs, (t,t+1), y0, t_eval = np.array([t+1]), max_step = 0.001)
    #t_eval = t, q me entrega, en q se evalua edo (puede evaluar en más puntos, xq eso lo define
    # inteligentemente solve_ivp internamente, solo q no me los entrega); ese t tiene q estar
    # dentro d (t,t+1) solve_ivp integra, i.e. obtiene sols en (t,t+1) y luego evalua eso
    # en t_eval.
    #Devuelve sol.y -> array [[T][L][M][I]]) , sol.t -> array [t]

    #Añadir resultados a array para gráficar
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

    print(f'sol.y: {sol.y[0]}')




##################################################################
######################   G R Á F I C O S   #######################
##################################################################

#### Resultados

T_count, L_count, M_count, I_count = sol_y_T, sol_y_L, sol_y_M, sol_y_I
t = sol_t
#print(f'I_count: {I_count}')
#print(f'Len: {len(I_count)}')
print(f'L_count: {L_count}')


#### Plots

## Plot

#plt.plot(t, T_count, 'D', color = 'm', label = 'T_count', markersize = 4)
#plt.plot(t, L_count, 'o',  color = 'y', label = 'L_count')
#plt.plot(t, M_count,'*', color = 'r', label = 'M_count')
#plt.plot(t, I_count,'x', color = 'c', label = 'I_count')



## Plot con cants normalizadas

# Normalización resultados

max_T_count = np.amax(T_count)  #Extrae máx valor del array #comprobado q fxna
T_count_norm = T_count / max_T_count #2do valor (máx q alcanza T) es un 1, bn
#print(f'max t: {max_T_count}')

max_L_count = np.amax(L_count)
#print("max L", max_L_count)
L_count_norm = L_count / max_L_count

max_M_count = np.amax(M_count)
M_count_norm = M_count / max_M_count

max_I_count = np.amax(I_count)
I_count_norm = I_count / max_T_count
##print(max_I_count/max_T_count)


#plt.plot(t, T_count_norm, 'D', color = 'green', label = 'T_count', markersize = 4)
plt.plot(t, L_count_norm, 'o',  color = 'c', label = 'L_count', markersize = 4)
#plt.plot(t, M_count_norm,'*', color = 'r', label = 'M_count')
#plt.plot(t, I_count_norm,'x', color = 'c', label = 'I_count')


#plt.ticklabel_format(style='plain') #desabilita la notación científica en eje Y


## RESULTADOS FINAL RAD

#######################################################
#L_sabado = L_sabado / np.amax(L_count)
#print(L_sabado)
########################################################


## Config gráfico

plt.grid()

plt.xlabel('Time (days)')
plt.ylabel('I cells count')
plt.title('L  v/s  time, frac = 15')
plt.legend()

#guardar lo dibujado hasta ahora
plt.savefig("L_DL_spor10_alphaL_0737.png", bbox_inches = 'tight')
#plt.savefig("T_15frac_rad_antes_ode.png", bbox_inches = 'tight')

#al final
plt.show()



#Extra



#for i, j in zip(T_alive_rkill, I_produced_rkill):
#   print(str(i) + " / " + str(j) + ' resta = ' + str(i+j) )


