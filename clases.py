import os
import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Clase que guarda la informacion de UNA serie DICOM (una carpeta con cortes)
class EstudioDicom:
    # constructor
    def __init__(self, carpeta, grupo, nombre):
        self.carpeta = carpeta
        self.grupo = grupo          # PPMI, Sarcoma, T2...
        self.nombre = nombre
        self.cortes = []            # matriz de pixeles de cada corte (original)
        self.filas = 0
        self.columnas = 0
        self.nombre_paciente = ''
        self.id_paciente = ''
        self.modalidad = ''
        self.voxel_alto = 0         # mm (PixelSpacing[0])
        self.voxel_ancho = 0        # mm (PixelSpacing[1])
        self.voxel_largo = 0        # mm (SliceThickness)
        self.matriz_pixeles = []    # volumen original (filas, columnas, cortes)
        self.volumen = []           # volumen con los cortes escalados al doble
        self.corte_corregido = []   # resultado de corregirAngulo
        self.angulo = 0
        self.corte_rotado = 0

        self.leer_carpeta()
        self.reconstruccion_3d()

    # Lee los .dcm de la carpeta y extrae la informacion del dataset
    def leer_carpeta(self):
        datasets = []
        numeros = []
        for a in os.listdir(self.carpeta):
            if a.lower().endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(self.carpeta, a))
                datasets.append(ds)
                numeros.append(int(ds.InstanceNumber))

        # se ordenan los cortes de menor a mayor numero de instancia
        Ind = np.argsort(numeros)

        # Informacion extraida directamente del dataset (primer corte)
        ds = datasets[Ind[0]]
        self.filas = int(ds.Rows)
        self.columnas = int(ds.Columns)
        self.nombre_paciente = str(ds.PatientName)
        self.id_paciente = str(ds.PatientID)
        self.modalidad = ds.Modality
        self.voxel_alto = float(ds.PixelSpacing[0])
        self.voxel_ancho = float(ds.PixelSpacing[1])
        self.voxel_largo = float(ds.SliceThickness)

        # Orientacion de los cortes (ImageOrientationPatient): si es axial
        # los planos reconstruidos se deben girar para que la cabeza quede vertical
        iop = ds.ImageOrientationPatient
        self.axial = abs(iop[0] * iop[4] - iop[1] * iop[3]) > 0.5
        # si el primer corte es el de mas abajo (z menor) hay que invertir el orden
        z_primero = float(datasets[Ind[0]].ImagePositionPatient[2])
        z_ultimo = float(datasets[Ind[-1]].ImagePositionPatient[2])
        self.invertir = z_primero < z_ultimo

        # matriz de pixeles de cada corte (solo los del mismo tamanho)
        for i in Ind:
            img = datasets[i].pixel_array
            if np.shape(img) == (self.filas, self.columnas):
                self.cortes.append(img)

        # volumen original (filas, columnas, cortes)
        self.matriz_pixeles = np.stack(self.cortes, axis=2)
    def reconstruccion_3d(self):
        cortes_escalados = []
        for corte in self.cortes:
            res = cv2.resize(corte, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cortes_escalados.append(res)
        self.volumen = np.stack(cortes_escalados, axis=2)

        # al escalar x2 el pixel en el plano mide la mitad
        self.voxel_alto_volumen = self.voxel_alto / 2
        self.voxel_ancho_volumen = self.voxel_ancho / 2

    # Rota un corte con centro en el punto medio de la imagen
    def corregirAngulo(self, grados, corte):
        img = self.volumen[:, :, corte]
        row, col = np.shape(img)

        # Matriz de rotacion
        MR = cv2.getRotationMatrix2D((col / 2, row / 2), grados, 1)
        # Rotacion: las esquinas vacias se rellenan con el valor minimo de la imagen
        # (en CT el fondo es negativo, si se rellena con 0 queda gris)
        fondo = int(np.min(img))
        self.corte_corregido = cv2.warpAffine(img, MR, (col, row), borderValue=fondo)
        self.angulo = grados
        self.corte_rotado = corte
        return self.corte_corregido
    def ver_informacion(self):
        print('Grupo                  ', self.grupo)
        print('Carpeta                ', self.carpeta)
        print('Nombre del paciente    ', self.nombre_paciente)
        print('ID del paciente        ', self.id_paciente)
        print('Modalidad              ', self.modalidad)
        print('Filas                  ', self.filas)
        print('Columnas               ', self.columnas)
        print('Numero de cortes       ', len(self.cortes))
        print('Voxel alto, ancho, largo (mm)  ', self.voxel_alto, self.voxel_ancho, self.voxel_largo)
        print('Tamanho matriz original        ', np.shape(self.matriz_pixeles))
        print('Tamanho volumen escalado       ', np.shape(self.volumen))
        print('Voxel volumen escalado (mm)    ', self.voxel_alto_volumen, self.voxel_ancho_volumen, self.voxel_largo)

    # Muestra los 3 planos del volumen
    def graficar_volumen(self):
        f, c, n = np.shape(self.volumen)

        # plano 1: corte del medio
        plano1 = self.volumen[:, :, n // 2]

        # plano 2 y 3: se redimensionan con cv2.resize para respetar los mm del voxel
        # (el espacio entre cortes es mayor que el tamanho del pixel)
        plano2 = self.volumen[f // 2, :, :]
        ancho2 = int(n * self.voxel_largo / self.voxel_ancho_volumen)
        plano2 = cv2.resize(plano2, (ancho2, c), interpolation=cv2.INTER_LINEAR)

        plano3 = self.volumen[:, c // 2, :]
        ancho3 = int(n * self.voxel_largo / self.voxel_alto_volumen)
        plano3 = cv2.resize(plano3, (ancho3, f), interpolation=cv2.INTER_LINEAR)

        # en imagenes axiales los cortes van de arriba a abajo: se transpone la matriz
        if self.axial:
            plano2 = plano2.T
            plano3 = plano3.T
            if self.invertir:
                plano2 = plano2[::-1, :]
                plano3 = plano3[::-1, :]

        if self.axial:
            # los planos son anchos y bajitos: el corte va a la izquierda
            # y los dos planos uno encima del otro a la derecha
            plt.figure(figsize=(15, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(plano1, cmap='gray')
            plt.title('Corte ' + str(n // 2))
            plt.axis('off')
            plt.subplot(2, 2, 2)
            plt.imshow(plano2, cmap='gray')
            plt.title('Plano fila ' + str(f // 2))
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(plano3, cmap='gray')
            plt.title('Plano columna ' + str(c // 2))
            plt.axis('off')
        else:
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(plano1, cmap='gray')
            plt.title('Corte ' + str(n // 2))
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(plano2, cmap='gray')
            plt.title('Plano fila ' + str(f // 2))
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(plano3, cmap='gray')
            plt.title('Plano columna ' + str(c // 2))
            plt.axis('off')
        plt.suptitle('Reconstruccion 3D - ' + self.nombre)
        plt.tight_layout()
        plt.show()
    def graficar_correccion(self):
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self.volumen[:, :, self.corte_rotado], cmap='gray')
        plt.title('Original')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(self.corte_corregido, cmap='gray')
        plt.title('Corregido ' + str(self.angulo) + ' grados')
        plt.axis('off')
        plt.show()

    # ================= PUNTOS 5 y 6 =================

    # Saca un corte del volumen (plano y numero 1, 2, 3...), lo normaliza a uint8
    # y corrige el aspect para que cada pixel mida lo mismo en alto y ancho
    def obtener_corte(self, volumen, plano, numero):
        # el volumen es (filas, columnas, cortes)
        if self.axial:
            ejes = {'transversal': 2, 'coronal': 0, 'sagital': 1}
        else:
            ejes = {'sagital': 2, 'transversal': 0, 'coronal': 1}

        if plano not in ejes:
            print('Plano no valido')
            return None, None
        eje = ejes[plano]
        total = volumen.shape[eje]
        if numero < 1 or numero > total:
            print('Corte no valido, debe estar entre 1 y', total)
            return None, None
        i = numero - 1

        # corte y tamanho en mm de sus filas y columnas (el volumen ya esta al doble)
        if eje == 2:
            corte = volumen[:, :, i]
            mm_fila = self.voxel_alto_volumen
            mm_col = self.voxel_ancho_volumen
        elif eje == 0:
            corte = volumen[i, :, :]
            mm_fila = self.voxel_ancho_volumen
            mm_col = self.voxel_largo
        else:
            corte = volumen[:, i, :]
            mm_fila = self.voxel_alto_volumen
            mm_col = self.voxel_largo

        # se recortan los valores extremos (1% mas oscuro y 0.5% mas brillante)
        # para que unos pocos pixeles muy brillantes no oscurezcan toda la imagen
        corte = corte.astype(np.float32)
        minimo = np.percentile(corte, 1)
        maximo = np.percentile(corte, 99.5)
        corte = np.clip(corte, minimo, maximo)

        # normalizar a uint8
        img = cv2.normalize(corte, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # aspect: se redimensiona igual que en graficar_volumen
        if abs(mm_col / mm_fila - 1) > 0.01:
            ancho_nuevo = int(img.shape[1] * mm_col / mm_fila)
            img = cv2.resize(img, (ancho_nuevo, img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # en axiales los cortes van de arriba a abajo: se transpone
        if self.axial and eje != 2:
            img = img.T
            if self.invertir:
                img = img[::-1, :]

        img = np.ascontiguousarray(img)
        return img, mm_fila     # ahora cada pixel mide mm_fila x mm_fila
