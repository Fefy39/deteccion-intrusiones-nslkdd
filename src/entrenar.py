"""Etapas 5 a 7 del pipeline: entrenamiento, evaluación y registro.

Uso:
    python -m src.entrenar --modelo xgboost
    python -m src.entrenar --modelo transformer --epocas 15
Guarda el modelo en models/, las probabilidades en resultados/ y una fila en
resultados/registro_experimentos.csv.
"""
import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import numpy as np

from src.config import MODELOS, RAIZ, SEMILLA
from src.datos_modelo import preparar
from src.evaluacion import metricas, registrar, resumen


def pesos_de_clase(y):
    """Pesos 'balanceados suavizados': raíz cuadrada del peso balanceado.
    Con el balanceado puro, U2R (42 ejemplos) pesaría ~480 veces más y el entrenamiento se vuelve inestable."""
    conteo = np.bincount(y, minlength=5)
    balanceado = len(y) / (5 * conteo)
    return {i: float(np.sqrt(p)) for i, p in enumerate(balanceado)}


def entrenar_xgboost(d, pesos):
    import xgboost as xgb
    modelo = xgb.XGBClassifier(objective="multi:softprob", n_estimators=1000, learning_rate=0.1,
                               max_depth=6, tree_method="hist", early_stopping_rounds=50,
                               eval_metric="mlogloss", random_state=SEMILLA, n_jobs=-1)
    w = np.array([pesos[c] for c in d["train"]["y"]])
    modelo.fit(d["train"]["tab"], d["train"]["y"], sample_weight=w,
               eval_set=[(d["val"]["tab"], d["val"]["y"])], verbose=100)
    modelo.save_model(MODELOS / "xgboost_v1.json")
    config = f"{modelo.best_iteration + 1} árboles, lr 0.1, prof. 6, pesos sqrt"
    return (modelo.predict_proba(d["val"]["tab"]), modelo.predict_proba(d["test"]["tab"]), config)


def entrenar_transformer(d, pesos, cards, epocas, version="v1", **hp):
    import keras
    from sklearn.metrics import f1_score
    from src.modelos import crear_ft_transformer
    keras.utils.set_random_seed(SEMILLA)
    modelo = crear_ft_transformer(d["train"]["num"].shape[1], cards, **hp)
    X = lambda p: [d[p]["num"], d[p]["cat"]]

    class F1Macro(keras.callbacks.Callback):
        """Calcula el F1 macro en validación al final de cada época."""
        def on_epoch_end(self, epoca, logs=None):
            pred = self.model.predict(X("val"), batch_size=2048, verbose=0).argmax(1)
            logs["val_f1_macro"] = f1_score(d["val"]["y"], pred, average="macro")
            print(f" - val_f1_macro: {logs['val_f1_macro']:.4f}")

    hist = modelo.fit(X("train"), d["train"]["y"], validation_data=(X("val"), d["val"]["y"]),
                      epochs=epocas, batch_size=512, class_weight=pesos, verbose=2,
                      callbacks=[F1Macro(), keras.callbacks.EarlyStopping(
                          monitor="val_f1_macro", mode="max", patience=3, restore_best_weights=True)])
    modelo.save(MODELOS / f"ft_transformer_{version}.keras")
    np.save(RAIZ / "resultados" / f"historial_transformer_{version}.npy", hist.history)
    config = (f"d={hp.get('d', 64)}, {hp.get('bloques', 3)} bloques, {hp.get('cabezas', 8)} cabezas, "
              f"dropout {hp.get('dropout', 0.1)}, {len(hist.history['loss'])} épocas")
    return (modelo.predict(X("val"), batch_size=2048, verbose=0),
            modelo.predict(X("test"), batch_size=2048, verbose=0), config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["xgboost", "transformer"], required=True)
    ap.add_argument("--epocas", type=int, default=15)
    args = ap.parse_args()

    MODELOS.mkdir(exist_ok=True); (RAIZ / "resultados").mkdir(exist_ok=True)
    d, cards, _ = preparar()
    pesos = pesos_de_clase(d["train"]["y"])
    print("Pesos de clase:", {k: round(v, 2) for k, v in pesos.items()})

    if args.modelo == "xgboost":
        p_val, p_test, config = entrenar_xgboost(d, pesos); nombre = "XGBoost"
    else:
        p_val, p_test, config = entrenar_transformer(d, pesos, cards, args.epocas); nombre = "FT-Transformer"

    np.savez_compressed(RAIZ / "resultados" / f"probs_{args.modelo}_v1.npz", val=p_val, test=p_test,
                        y_val=d["val"]["y"], y_test=d["test"]["y"], label_test=d["test"]["label"])
    m_val, m_test = metricas(d["val"]["y"], p_val.argmax(1)), metricas(d["test"]["y"], p_test.argmax(1))
    resumen(f"{nombre} - validación", m_val); resumen(f"{nombre} - prueba (KDDTest+)", m_test)
    registrar(nombre, "v1", config, m_val, m_test)


if __name__ == "__main__":
    main()
