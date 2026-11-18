"""Pruebas del pipeline que GitHub Actions ejecuta en cada cambio."""
import pandas as pd
from src.config import CATEGORIA_ATAQUE, CLASES, COLUMNAS
from src.preprocesamiento import a_cinco_clases, construir_preprocesador, limpiar, transformar_sesgadas


def fila(label="normal", **k):
    base = {c: 0 for c in COLUMNAS}
    base.update(protocol_type="tcp", service="http", flag="SF", label=label, difficulty=20)
    base.update(k)
    return base


def test_esquema_tiene_43_columnas():
    assert len(COLUMNAS) == 43


def test_todas_las_categorias_son_validas():
    assert set(CATEGORIA_ATAQUE.values()) == set(CLASES)


def test_mapeo_a_cinco_clases():
    df = pd.DataFrame([fila("normal"), fila("neptune"), fila("satan"), fila("guess_passwd"), fila("rootkit")])
    assert a_cinco_clases(df).tolist() == [0, 1, 2, 3, 4]


def test_limpieza_quita_duplicados_y_columnas():
    df = pd.DataFrame([fila(), fila()])
    out = limpiar(df)
    assert len(out) == 1
    assert "difficulty" not in out and "num_outbound_cmds" not in out


def test_log_en_columnas_sesgadas():
    df = pd.DataFrame([fila(src_bytes=0), fila(src_bytes=1_000_000)])
    out = transformar_sesgadas(df)
    assert out["src_bytes"].max() < 20


def test_preprocesador_ignora_servicio_desconocido():
    df = limpiar(pd.DataFrame([fila(service="http"), fila(service="ftp")]))
    prep = construir_preprocesador(df).fit(df)
    nuevo = limpiar(pd.DataFrame([fila(service="servicio_nuevo")]))
    assert prep.transform(nuevo).shape[1] == prep.transform(df).shape[1]
