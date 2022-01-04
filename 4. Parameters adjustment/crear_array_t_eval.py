import numpy as np


def crear_array_t_eval(dia_actual: int, t_medido: np.array) -> np.array:
    '''Crea un array con todos los tiempos que están dentro de dia_actual'''
    ## Funciona bien.

    array_t_eval = np.array([])

    # Buscamos si se contaron empíricamente los linfocitos en el "día actual"
    for t_aux in t_medido:

        if int(t_aux) == dia_actual:

            # Si contaron empíricamente L en el día actual. Entonces:
            # Es necesario que se obtenga la solución de la ODE para aquellos tiempos
            # que están contenidos en el "día_actual". Para ello generamos el array
            # que se le entregara al keyword "t_eval=" en solve_ivp.

            array_t_eval = np.append(array_t_eval, t_aux)
                # obs: t_aux es un entero, no un array.
                # Esto no es un problema para np.append. i.e puede unir un array con un float/int


    return array_t_eval



if __name__ == "__main__":

    t_medido_aux = np.array([3, 3.3, 5.1, 5.7, 5.9, 6.7, 9])

    v_array_t_eval = crear_array_t_eval(5, t_medido_aux)

    print(v_array_t_eval)






