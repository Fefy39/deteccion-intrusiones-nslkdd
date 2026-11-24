"""Métricas de evaluación: recall por clase, F1 macro, matriz de confusión y registro."""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src.config import CLASES, RAIZ

NOMBRES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def metricas(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    ataque_real, ataque_pred = y != 0, pred != 0
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", labels=range(5), zero_division=0),
        "recall": dict(zip(NOMBRES, r)), "precision": dict(zip(NOMBRES, p)), "f1": dict(zip(NOMBRES, f)),
        # Vista binaria (¿se detectó que era un ataque, sin importar el tipo?)
        "deteccion_ataques": ataque_pred[ataque_real].mean(),
        "falsas_alarmas_x1000": 1000 * ataque_pred[~ataque_real].mean(),
        "matriz": confusion_matrix(y, pred, labels=range(5)),
    }


def resumen(nombre, m):
    print(f"\n=== {nombre} ===")
    print(f"Accuracy {m['accuracy']:.4f} | F1 macro {m['f1_macro']:.4f} | "
          f"Ataques detectados {m['deteccion_ataques']:.4f} | Falsas alarmas x1000 {m['falsas_alarmas_x1000']:.1f}")
    print("Recall por clase: " + ", ".join(f"{k} {v:.3f}" for k, v in m["recall"].items()))


def registrar(modelo, version, config, m_val, m_test, ruta=RAIZ / "resultados" / "registro_experimentos.csv"):
    ruta = Path(ruta); ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists()
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "modelo", "version", "configuracion", "f1_macro_val", "f1_macro_test",
                        "accuracy_test", "deteccion_ataques_test", "falsas_alarmas_x1000_test",
                        "recall_r2l_test", "recall_u2r_test"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), modelo, version, config,
                    round(m_val["f1_macro"], 4), round(m_test["f1_macro"], 4), round(m_test["accuracy"], 4),
                    round(m_test["deteccion_ataques"], 4), round(m_test["falsas_alarmas_x1000"], 1),
                    round(m_test["recall"]["R2L"], 4), round(m_test["recall"]["U2R"], 4)])
