import pandas as pd
import numpy as np

def main(): #yo practicando con pandas :)

    excel_file = 'DVH acumulado - pares x,y.xlsx'
    Lrkill = pd.read_excel(excel_file, sheet_name= 1)

    #print(Lrkill.head())

    df = pd.DataFrame(Lrkill) #formato para trabajar cn los datos
    cols = [0] #columnas que quiero extraer
    Lcirculantes = df[df.columns[cols]]
    #print(Lcirculantes)


    #tmbn puedo seleccionar la columa directamente por el nombre
    Lcirculantes_2 = df["Linfocitos Circulantes %"]
    #print(Lcirculantes_2) #sin header (título columna)


    #print(Lcirculantes_2[0])

    #print(Lcirculantes.shape[0])

    return


def rad_linfo(name_excel,cols,Ln, alpha_L):
    '''El único cambio con r/Pandas_y_linfocitos.py es el valor de alpha_L'''

    #cols: lista con n° columnas del excel a extraer (inicia con 0)
    #excel entrega %L_circulantes, Dosis en Gy.

    ## PARA EL AJUSTE DE PARÁMETROS #######
        ## Recomendación ignacio 04/01/2022
        # dejar alpha_L con un valor un poco mayor o igual a alpha_T
            # mjr igual al ppio.

    print()
    Lnplusone = 0
    Lrkill = pd.read_excel(name_excel, sheet_name = 1) #abrir excel, página 1 (cuenta desde 0)
    df = pd.DataFrame(Lrkill) # Formato para trabajar cn los datos
                              # DataFrame is a 2-dimensional labeled data structure

    #df.columns: saca headers de las cols(lista) pedidas
    dvh_linfo_rkill = df[df.columns[cols]] #saca header(título columna)+ columnas(datos);
                                        # tmbn puede ir nombre columna en lugar de df.columns[cols]

    #Formato dataframe al printear:
        #    col1 col2
        # 0    a    d
        # 1    b    h


    #COMENTARIO: extrae bien la columna de linfocitos circulantes :)

    #dvh_linfo_rkill.shape: contiene tantas 'listas'(arrays) de n elementos (n = n° columas extraidas)
                          # como sea el largo de las columnas excel:  (largo columna, n)
                          # largo columna inicia en 0 y NO incluye header

    #dvh_linfo_rkill.loc[fila][columna]: entrega elemento en fila y columna señaladas (printea un dato s/header)
                            # solo [fila], extrae fila, i.e. array de n elementos(n° columas extraidas)
                            # (printea columna de cosas cn formato"header: dato")

            # .loc[0][0]
                # primer [0] saca fila zero; segundo [0] saca primer(pos 0) elemento en fila zero.
                # De no poner el segundo [0] va a llamar a la primera fila con el header de cada columna.


    for i in range(dvh_linfo_rkill.shape[0]):

        # L_circulantes viene en %, hay que pasarlo a decimal
        Lni = Ln * (dvh_linfo_rkill.loc[i][0] * 0.01)
            #dvh_linfo_rkill.loc[i][0] extraigo primer elemento '[0]' de la fila 'i'


        # Dosis viene %Dosis*D_T con D_T=58/15. i.e. viene como DL, para cada i, listo para usar.
        DLi = dvh_linfo_rkill.loc[i][1]*10.2        # afectados por una dosis rad DLi
                                #se ponderó *10 para asemejar a curva del paper.
            #dvh_linfo_rkill.loc[i][1] extraigo segundo elemento '[1]' de la fila 'i'

        L_aux = Lni * np.exp( - alpha_L * DLi )


        Lnplusone += L_aux

    Lnplusone += Ln * (1 - 49.5129250 * 0.01) # Se agregan los linfocitos cuyo DLi = 0; que corresponde al tot

    return Lnplusone
    #Se comprobó por excel que efectivamente da 2734762016.9819927 el Ln+1 dsp de la primera rad.



#print(rad_linfo('DVH acumulado - pares x,y.xlsx',[0,1],5.61*10**9))


if __name__ == '__main__':
    #use this block to allow or prevent parts of code from being run when the modules are imported
    main()