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