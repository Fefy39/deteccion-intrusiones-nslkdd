"""Prepara los datos en el formato que necesita cada modelo.

- FT-Transformer: variables numéricas normalizadas + variables categóricas como índices
  (cada una tendrá su propio embedding).
- XGBoost: las mismas numéricas + one-hot de las categóricas (121 columnas, como en el Avance II).

Se usa exactamente la misma división del Avance II (80/20 estratificado, semilla 42).
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import CATEGORICAS, SEMILLA
from src.preprocesamiento import (a_cinco_clases, cargar, limpiar, transformar_sesgadas,
                                  validar)


def preparar():
    train, test = cargar("train"), cargar("test")
    validar(train); validar(test)
    train = transformar_sesgadas(limpiar(train))
    test = transformar_sesgadas(limpiar(test))
    y_full, y_test = a_cinco_clases(train).values, a_cinco_clases(test).values

    df_tr, df_val, y_tr, y_val = train_test_split(
        train, y_full, test_size=0.2, stratify=y_full, random_state=SEMILLA)

    numericas = [c for c in train.columns if c not in CATEGORICAS + ["label"]]
    escalador = StandardScaler().fit(df_tr[numericas])          # solo entrenamiento
    # Las categorías nuevas (no vistas al entrenar) se mapean al índice -1 -> se suma 1 = 0
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(df_tr[CATEGORICAS])
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df_tr[CATEGORICAS])

    def vista(df):
        num = escalador.transform(df[numericas]).astype("float32")
        cat = (ordinal.transform(df[CATEGORICAS]) + 1).astype("int32")   # 0 = desconocida
        tab = np.hstack([num, onehot.transform(df[CATEGORICAS])]).astype("float32")
        return num, cat, tab

    cardinalidades = [len(c) + 1 for c in ordinal.categories_]          # +1 por "desconocida"
    datos = {}
    for nombre, df, y in [("train", df_tr, y_tr), ("val", df_val, y_val), ("test", test, y_test)]:
        num, cat, tab = vista(df)
        datos[nombre] = {"num": num, "cat": cat, "tab": tab, "y": y.astype("int32"),
                         "label": df["label"].values}
    return datos, cardinalidades, numericas
