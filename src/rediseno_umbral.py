"""Rediseño v2: regla de alerta según la probabilidad de "normal".

En la versión v1 la clase final es la de mayor probabilidad (argmax). Como dejar pasar un ataque
es el error más grave, en v2 se genera una alerta siempre que el modelo NO esté muy seguro de que
la conexión es normal: si P(normal) < umbral, se asigna la clase de ataque más probable.

El umbral se elige SOLO con validación, con una regla de negocio: el máximo umbral que mantenga
las falsas alarmas en validación en 5 o menos por cada 1.000 conexiones normales.
Uso: python -m src.rediseno_umbral --modelo xgboost
"""
import argparse
import numpy as np
from src.config import RAIZ
from src.evaluacion import metricas, registrar, resumen

UMBRALES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
LIMITE_FALSAS_ALARMAS = 5.0   # por cada 1.000 conexiones normales (en validación)


def aplicar_umbral(p, umbral):
    pred = p.argmax(1)
    alerta = p[:, 0] < umbral
    pred[alerta] = p[alerta, 1:].argmax(1) + 1      # la clase de ataque más probable
    pred[~alerta] = 0
    return pred


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modelo", required=True); a = ap.parse_args()
    z = np.load(RAIZ / "resultados" / f"probs_{a.modelo}_v1.npz", allow_pickle=True)
    filas, elegido = [], 0.5
    for u in UMBRALES:
        mv = metricas(z["y_val"], aplicar_umbral(z["val"], u))
        mt = metricas(z["y_test"], aplicar_umbral(z["test"], u))
        filas.append((u, mv["falsas_alarmas_x1000"], mt["deteccion_ataques"], mt["falsas_alarmas_x1000"], mt["f1_macro"]))
        if mv["falsas_alarmas_x1000"] <= LIMITE_FALSAS_ALARMAS:
            elegido = u
    print("umbral | FA val x1000 | detección test | FA test x1000 | F1 macro test")
    for f in filas: print(" | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in f))
    print("Umbral elegido con validación:", elegido)
    np.save(RAIZ / "resultados" / f"umbral_{a.modelo}.npy", np.array(filas))
    mv = metricas(z["y_val"], aplicar_umbral(z["val"], elegido))
    mt = metricas(z["y_test"], aplicar_umbral(z["test"], elegido))
    resumen(f"{a.modelo} v2 - validación", mv); resumen(f"{a.modelo} v2 - prueba", mt)
    nombre = "XGBoost" if a.modelo == "xgboost" else "FT-Transformer"
    registrar(nombre, "v2", f"v1 + alerta si P(normal) < {elegido}", mv, mt)


if __name__ == "__main__":
    main()
