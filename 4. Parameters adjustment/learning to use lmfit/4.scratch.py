import numpy as np

def f(a,b,*kk):
    ex = (a,b)
    print(a,b)
    return

f("s","e")


model = [1,2,3,4,5,6,7,8]
#print(model[:,1])


print(type(2.))
print(type(2))

lista = [10, 5, 1, 11, 20, 44]
print(lista)
lista.sort()
print(f'2da: {lista}')

T0 = 1.07 * 10 ** 11
L0 = 5.61 * 10 ** 9
M0 = 1.07 * 10 ** 8    # Patient with metastasis
I0 = 0
y0 = np.array([T0, L0, M0, I0])
print(y0[1])