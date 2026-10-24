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