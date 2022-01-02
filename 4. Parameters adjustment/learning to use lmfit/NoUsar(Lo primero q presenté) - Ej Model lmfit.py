import numpy as np
import lmfit
import matplotlib.pyplot as plt


##### lmfit
## Non-Linear Least-Squares Minimization and Curve-Fitting for Python
## Model -> automatically generates the appropriate residual function
## minimize _> la fx debe retornar the array to be minimized, i.e. cn los residuos
#               xq eso es lo q se minimiza

##the basics

#example

# Vo = Vmax / (1 + K / s) -> función a simular
import numpy as np


def xpoints_and_parameters(x, K, V): #función que simula in/outout
    return V / (1 + K / x) #y

s = np.array([2.5, 5, 10, 15, 20]) #x points
vo = np.array([0.024, 0.036, 0.053, 0.060, 0.064]) #y points


#importamos clase Model que permite usar como modelo la función creada
mod = lmfit.Model(xpoints_and_parameters)
        #obs -> Debe ser con mayúscula. hay otro con minúscula, ese NO es la clase.

result = mod.fit(vo, x = s, K = 1, V = 1) #AQUÍ VA 'method= powell'
    # fit(ydata, xdata =, io_guesses for parameters {1 for default} )
    # .fit() -> retorna ModelResult Object. -> ese si se puede plottear cn matplotlib
    #        -> subclase de Minimizer

print(result.best_values) #entrega parámetros fitteados
#print(result.best_values["K"])




    #contains fit statistics and best-fit values with uncertainties and correlations.
#print(result.fit_report())


    #Forma alternativa de calcular los intervalos de confianza
#ModelResult.conf_interval(**kwargs)


#tuve q importar importlib-metadata.
result.plot() # Plot the fit results (¿best fit') and residuals using matplotlib.
#numponts = 100
#will produce a matplotlib figure!
#obs: saqué shot_init = True, q no sé q es, del argumento en la fx anterior


#Si quiero plotter solo:
    #best fit:
#plt.plot(s, result.best_fit, 'r-', label = 'best fit')

    #initial fit
#plt.plot(s, result.init_fit, 'k--', label = 'initial fit')

    #puntos a fittear
#plt.plot(s, vo, 'bo')


plt.grid()
plt.legend()
plt.show()
























