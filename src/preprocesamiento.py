"""Etapas 2 a 4 del pipeline: limpieza/validación, preprocesamiento y división.

Uso: python -m src.preprocesamiento
Genera en data/processed/: X_train, X_val, X_test, y_train, y_val, y_test (.npz),
y en models/: preprocesador.joblib (para usar exactamente la misma transformación en la API).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (ARCHIVOS, CATEGORIA_ATAQUE, CATEGORICAS, CLASES, COLUMNAS,
                        DATOS_PROC, DATOS_RAW, MODELOS, SEMILLA)


def cargar(nombre: str) -> pd.DataFrame:
    return pd.read_csv(DATOS_RAW / ARCHIVOS[nombre], header=None, names=COLUMNAS)


def validar(df: pd.DataFrame) -> None:
    """Valida el esquema: 43 columnas, sin nulos y etiquetas conocidas."""
    assert list(df.columns) == COLUMNAS, "Las columnas no coinciden con el esquema"
    assert df.isna().sum().sum() == 0, "Hay valores nulos"
    desconocidas = set(df["label"]) - set(CATEGORIA_ATAQUE)
    assert not desconocidas, f"Etiquetas sin categoría: {desconocidas}"


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Quita duplicados, la columna de dificultad y columnas sin variación."""
    df = df.drop_duplicates().drop(columns=["difficulty"])
    return df.drop(columns=["num_outbound_cmds"])  # siempre vale 0


SESGADAS = ["duration", "src_bytes", "dst_bytes"]


def transformar_sesgadas(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica log(1 + x) a columnas muy sesgadas (p. ej., src_bytes llega a 1.379 millones)."""
    df = df.copy()
    df[SESGADAS] = np.log1p(df[SESGADAS])
    return df


def a_cinco_clases(df: pd.DataFrame) -> pd.Series:
    return df["label"].map(CATEGORIA_ATAQUE).map({c: i for i, c in enumerate(CLASES)})


def construir_preprocesador(df: pd.DataFrame) -> ColumnTransformer:
    numericas = [c for c in df.columns if c not in CATEGORICAS + ["label"]]
    return ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])


def ejecutar() -> dict:
    train, test = cargar("train"), cargar("test")
    validar(train); validar(test)
    train, test = limpiar(train), limpiar(test)
    train, test = transformar_sesgadas(train), transformar_sesgadas(test)
    y_full, y_test = a_cinco_clases(train), a_cinco_clases(test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        train, y_full, test_size=0.2, stratify=y_full, random_state=SEMILLA)

    prep = construir_preprocesador(train)
    X_tr = prep.fit_transform(X_tr)          # se ajusta SOLO con entrenamiento
    X_val = prep.transform(X_val)
    X_te = prep.transform(test)

    DATOS_PROC.mkdir(parents=True, exist_ok=True); MODELOS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATOS_PROC / "datos.npz", X_train=X_tr, X_val=X_val, X_test=X_te,
                        y_train=y_tr.values, y_val=y_val.values, y_test=y_test.values)
    joblib.dump(prep, MODELOS / "preprocesador.joblib")

    resumen = {"train": X_tr.shape, "val": X_val.shape, "test": X_te.shape}
    print(json.dumps({k: list(v) for k, v in resumen.items()}, indent=2))
    return resumen


if __name__ == "__main__":
    ejecutar()
