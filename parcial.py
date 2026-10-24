#PARCIAL 3 
#MARIA PAULA
#SAMUEL VELEZ PEREZ
#GRUPO 2
# ============================================================
# LectorCarpetas (versión simple)
# Busca archivos DICOM en carpetas y subcarpetas, y muestra la información ordenada.
# Librerías de clase: os, numpy, pydicom
# ============================================================
import os
import warnings
import numpy as np
from pydicom import dcmread

warnings.filterwarnings("ignore")   # oculta los avisos amarillos de pydicom (son inofensivos)


class LectorCarpetas:

    def __init__(self, carpetas):
        # carpetas: una ruta (str) o una lista de rutas
        #   LectorCarpetas("datos")
        #   LectorCarpetas(["Sarcoma", "series", "PPMI", "T2"])
        if isinstance(carpetas, str):
            carpetas = [carpetas]
        self.carpetas = carpetas # {nombre de la serie: lista de cortes (Dataset) ordenados}

    # ------------------------------------------------------------
    # 1) Buscar todos los DICOM con imagen en las carpetas y subcarpetas
    # ------------------------------------------------------------
    def buscarSeries(self):
        self.series = {}
        ignorados = 0

        for carpeta in self.carpetas:
            raiz = os.path.dirname(os.path.normpath(carpeta))     # para que el nombre empiece en Sarcoma, PPMI...

            # os.walk entra a todas las subcarpetas, sin importar cuántos niveles haya
            for ruta_actual, subcarpetas, archivos in os.walk(carpeta):
                for archivo in archivos:
                    try:
                        ds = dcmread(os.path.join(ruta_actual, archivo))
                    except Exception:
                        ignorados += 1               # no es un archivo DICOM
                        continue
                    if "PixelData" not in ds:
                        ignorados += 1               # es DICOM pero no tiene imagen
                        continue

                    nombre = os.path.relpath(ruta_actual, raiz)
                    # Si en la misma carpeta hay otra serie distinta, se separa con (2), (3)...
                    nombre = self._nombreLibre(nombre, ds)
                    self.series.setdefault(nombre, []).append(ds)

        for nombre in self.series:
            self.series[nombre] = self._ordenar(self.series[nombre])

        print("Series encontradas:", len(self.series), "| archivos ignorados:", ignorados)

    def _nombreLibre(self, nombre, ds):
        # Devuelve el nombre de la serie a la que pertenece este corte
        uid = str(ds.get("SeriesInstanceUID", ""))
        contador = 1
        nombre_probado = nombre
        while nombre_probado in self.series:
            uid_guardado = str(self.series[nombre_probado][0].get("SeriesInstanceUID", ""))
            if uid == uid_guardado:
                return nombre_probado                 # es la misma serie
            contador += 1
            nombre_probado = nombre + " (" + str(contador) + ")"
        return nombre_probado

    # ------------------------------------------------------------
    # 2) Ordenar los cortes de una serie
    # ------------------------------------------------------------
    def _ordenar(self, lista):
        # Intento 1: por la posición en el espacio (la coordenada x, y o z que más cambie)
        posiciones = [d.get("ImagePositionPatient", None) for d in lista]
        if None not in posiciones and len(lista) > 1:
            posiciones = np.array([[float(v) for v in p] for p in posiciones])
            eje = int(np.argmax(posiciones.max(axis=0) - posiciones.min(axis=0)))
            lista.sort(key=lambda d: float(d.ImagePositionPatient[eje]))
        # Intento 2: por InstanceNumber, solo si todos lo tienen
        elif all(d.get("InstanceNumber", None) is not None for d in lista):
            lista.sort(key=lambda d: int(d.InstanceNumber))
        # Intento 3: por nombre de archivo
        else:
            lista.sort(key=lambda d: str(d.filename))
        return lista

    # ------------------------------------------------------------
    # 3) Mostrar la información ordenada
    # ------------------------------------------------------------
    def listarSeries(self):
        # Tabla con todas las series encontradas
        print()
        print(f"{'N°':<4}{'SERIE':<30}{'CORTES':<8}{'TAMAÑO':<12}{'TIPO':<6}PACIENTE")
        print("-" * 85)
        for numero, nombre in enumerate(self.series):
            ds = self.series[nombre][0]
            tamano = str(ds.Rows) + "x" + str(ds.Columns)
            print(f"{numero:<4}{nombre[-28:]:<30}{len(self.series[nombre]):<8}{tamano:<12}"
                  f"{str(ds.get('Modality', '?')):<6}{str(ds.get('PatientName', '?'))[:25]}")
        print()

    def infoSerie(self, serie):
        # Ficha detallada de una serie (se puede pedir por número o por nombre)
        nombre = self._buscarNombre(serie)
        if nombre is None:
            return
        ds = self.series[nombre][0]
        spacing = [float(v) for v in ds.get("PixelSpacing", [1, 1])]
        grosor = float(ds.get("SliceThickness", 1))

        print("=" * 45)
        print("SERIE:", nombre)
        print("=" * 45)
        print("Paciente      :", ds.get("PatientName", "?"))
        print("Modalidad     :", ds.get("Modality", "?"))
        print("Filas         :", ds.Rows)
        print("Columnas      :", ds.Columns)
        print("Cortes        :", len(self.series[nombre]))
        print("Voxel (mm)    :", spacing[0], "x", spacing[1], "x", grosor, " (alto x largo x ancho)")
        print("-" * 45)

    # ------------------------------------------------------------
    # 4) Construir el volumen 3D
    # ------------------------------------------------------------
    def _buscarNombre(self, serie):
        # Acepta el número de la tabla (0, 1, 2...) o el nombre de la serie
        if isinstance(serie, int):
            if 0 <= serie < len(self.series):
                return list(self.series.keys())[serie]
            print("Número fuera de rango. Hay", len(self.series), "series (0 a", len(self.series) - 1, ")")
            return None
        if serie in self.series:
            return serie
        print("Serie no encontrada. Usa el número de la tabla (listarSeries).")
        return None

    def construirVolumen(self, serie):
        # Devuelve: volumen (cortes, filas, columnas), pixel_spacing y grosor
        nombre = self._buscarNombre(serie)
        if nombre is None:
            return None, None, None

        cortes = self.series[nombre]
        volumen = np.stack([d.pixel_array for d in cortes]).astype(np.int16)

        spacing = [float(v) for v in cortes[0].get("PixelSpacing", [1, 1])]
        grosor = float(cortes[0].get("SliceThickness", 1))
        return volumen, spacing, grosor

# ============================================================
# EJEMPLO DE USO
# ============================================================
if __name__ == "__main__":
    lector = LectorCarpetas(["Sarcoma", "series", "PPMI", "T2"])
    lector.buscarSeries()
    lector.listarSeries()
    lector.infoSerie(0)
    volumen, pixel_spacing, grosor = lector.construirVolumen(0)
    print("Volumen:", volumen.shape)

