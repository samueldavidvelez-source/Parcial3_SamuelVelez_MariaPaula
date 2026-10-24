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