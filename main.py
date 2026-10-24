from clases import GestorDicom, pedir_numero


# Pide el plano con un numero para evitar errores al escribir
def pedir_plano():
    print('1. transversal')
    print('2. coronal')
    print('3. sagital')
    n = pedir_numero('Seleccione el plano: ')
    if n == 1:
        return 'transversal'
    if n == 2:
        return 'coronal'
    if n == 3:
        return 'sagital'
    print('Plano no valido')
    return None


def main():
    gestor = GestorDicom()

    while True:
        print('\n========= MENU - Analisis de imagenes DICOM =========')
        print('1. Cargar una carpeta DICOM')
        print('2. Cargar todas las carpetas de "datos" (PPMI, Sarcoma, T2)')
        print('3. Listar estudios cargados')
        print('4. Ver informacion de un estudio')
        print('5. Ver reconstruccion 3D')
        print('6. Corregir angulo')
        print('7. Ecualizar un corte (histogramas)')
        print('8. Segmentar elementos de interes (circulo y area)')
        print('9. Salir')
        opcion = input('Seleccione una opcion: ')

        if opcion == '1':
            carpeta = input('Ruta de la carpeta: ')
            gestor.cargar_carpeta(carpeta)

        elif opcion == '2':
            gestor.cargar_datos('datos')
            gestor.listar()

        elif opcion == '3':
            gestor.listar()

        elif opcion == '4':
            estudio = gestor.seleccionar()
            if estudio is not None:
                estudio.ver_informacion()

        elif opcion == '5':
            estudio = gestor.seleccionar()
            if estudio is not None:
                estudio.graficar_volumen()

        elif opcion == '6':
            estudio = gestor.seleccionar()
            if estudio is not None:
                grados = float(input('Angulo de inclinacion en grados: '))
                total = estudio.volumen.shape[2]
                corte = pedir_numero('Numero de corte (1 a ' + str(total) + '): ')
                if corte < 1 or corte > total:
                    corte = total // 2 + 1
                    print('Corte no valido, se usa el corte de la mitad: ', corte)
                estudio.corregirAngulo(grados, corte - 1)
                estudio.graficar_correccion()

        elif opcion == '7':
            estudio = gestor.seleccionar()
            if estudio is not None:
                plano = pedir_plano()
                if plano is not None:
                    numero = pedir_numero('Numero de corte (empieza en 1): ')
                    estudio.ecualizar_corte(plano, numero)

        elif opcion == '8':
            estudio = gestor.seleccionar()
            if estudio is not None:
                plano = pedir_plano()
                if plano is not None:
                    numero = pedir_numero('Numero de corte (empieza en 1): ')
                    kernel = pedir_numero('Tamanho del kernel (ej. 3, 5, 7): ')
                    iteraciones = pedir_numero('Iteraciones (ej. 1, 2): ')
                    if kernel < 1 or iteraciones < 1:
                        print('Kernel e iteraciones deben ser mayores que 0')
                    else:
                        estudio.segmentar(plano, numero, kernel, iteraciones)

        elif opcion == '9':
            print('Fin del programa')
            break

        else:
            print('Opcion no valida')


if __name__ == '__main__':
    main()