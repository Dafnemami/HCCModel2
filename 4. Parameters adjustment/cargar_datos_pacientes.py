import pandas as pd
import numpy as np

#########################################################
#             FX CARGAR DATOS PACIENTES                 #
#########################################################

# Para que funcion

def cargar_datos_pacientes(name_excel):
    '''
    Esta función carga desde un archivo excel, todos los conteos de linfocitos
    de uno (P.o varios) pacientes.

    :param cols: lista con n° columnas del excel a extraer (inicia con 0)
    :return: dos arrays, uno con n° L durante y dsp rad de uno/varios pacientes y otro con el t
    '''

    # POR EL MOMENTO, ESTO ES PARA UN PACIENTE, MODELABLE PARA MÁS.

    ### un paciente

    datos = pd.read_excel(name_excel, sheet_name=1) #abrir excel, página 1 dnd ta fig 3a
    df_datos = pd.DataFrame(datos) # Formato para trabajar cn los datos
    # DataFrame is a 2-dimensional labeled data structure

    #df.columns[(lista)]: saca headers de las columnas entregando su índice #cuenta desde 0
    cols_df_datos = df_datos[df_datos.columns[[0,1]]] # tiempo, datos en columna 0 y 1


    ######## SOBRE COMO TRABAJAR CON LAS COLUMNAS EXTRAIDAS ######################

    # .shape: contiene m arrays de n elementos (m= n°filas, n=n°columnas extridas)
    # - arroja (m,n)

    # .loc[fila][columna]: entrega elemento en fila y columna (printea dato s/header)
    # solo [fila], extrae fila, i.e. array de n elementos(n° columas extraidas)
    # (printea columna de cosas cn formato"header: dato")

    # .loc[0][0]: primer [0] saca fila zero; segundo [0] saca primer(pos 0) elemento en fila zero.
    # De no poner el segundo [0] va a llamar a la primera fila con el header de cada columna.

    ###############################################################################


    # LUEGO Q LEEMOS LOS DATOS DE TODOS LOS PACIENTES EN UN FORMATO CONVENIENTE Q AÚN NO SÉ,
    # PROCEDEMOS A ORDENAR DE FORMA ASCENDENTE EL TIEMPO, Y EN BASE A ESO LOS DATOS.

    lista_tuplas_aux = [] #cada tupla (dato, tiempo)
    tiempo_aux = [] #esta lista me permitirá ordenar los datos en base al tiempo.
    for fila in range(cols_df_datos.shape[0]): #extra numero de filas

        tupla_aux = (cols_df_datos.loc[fila][0], cols_df_datos.loc[fila][1]) #t, dato

        #agrego la tupla a la lista auxiliar y el tiempo a la lista 'tiempo_aux'
        lista_tuplas_aux.append(tupla_aux)
        tiempo_aux.append(cols_df_datos.loc[fila][0]) #t va en la columna 0

    #print(f'lista_tupla_aux FUERA\n {lista_tuplas_aux}')

    tiempo_aux.sort() #ordena del tiempo en forma ascendente

    lista_tuplas = []
    for t_aux in tiempo_aux:
        for par in lista_tuplas_aux:
            if par[0] == t_aux:
                lista_tuplas.append(par)
                lista_tuplas_aux.remove(par)
                # Si hay dos puntos q ocurren al mismo tiempo, se añadirán sin problema

    #print(lista_tuplas) #probar q los datos están ordenados.

    tiempo, datos = zip(*lista_tuplas)
    #print(lista_tuplas)
    #print(len(lista_tuplas))

    return tiempo, datos



