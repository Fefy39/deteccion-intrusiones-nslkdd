"""Etapa 1 del pipeline: recolección de datos.

Descarga KDDTrain+ y KDDTest+ (NSL-KDD) y los guarda en data/raw/.
Uso: python -m src.descargar_datos
"""
import urllib.request
from src.config import ARCHIVOS, DATOS_RAW, URL_BASE


def descargar(forzar: bool = False) -> None:
    DATOS_RAW.mkdir(parents=True, exist_ok=True)
    for nombre in ARCHIVOS.values():
        destino = DATOS_RAW / nombre
        if destino.exists() and not forzar:
            print(f"Ya existe: {destino.name}")
            continue
        print(f"Descargando {nombre} ...")
        urllib.request.urlretrieve(URL_BASE + nombre, destino)
        print(f"Guardado en {destino}")


if __name__ == "__main__":
    descargar()
