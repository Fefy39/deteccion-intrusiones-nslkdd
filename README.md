# Detección de intrusiones en redes con deep learning (NSL-KDD)

Proyecto final del curso **Aplicaciones en Deep Learning** – Especialización en Inteligencia Artificial, Universidad de Cundinamarca.
Autora: Jenifer Cárdenas Aguilera · Docente: Wilson Mauro Rojas Reales

## Problema
Clasificar cada conexión de red como **normal** o como uno de cuatro tipos de ataque (**DoS, Probe, R2L, U2R**), priorizando que ningún ataque pase sin ser detectado (recall por clase de ataque y F1 macro).

## Datos
NSL-KDD (Canadian Institute for Cybersecurity): KDDTrain+ (125.973 conexiones) y KDDTest+ (22.544), 41 características.
Los datos no se suben al repositorio; se descargan con el pipeline.

## Pipeline
| Etapa | Script | Qué hace |
|---|---|---|
| 1. Recolección | `src/descargar_datos.py` | Descarga KDDTrain+ y KDDTest+ en `data/raw/` |
| 2. Limpieza y validación | `src/preprocesamiento.py` | Valida el esquema, nulos, etiquetas y duplicados; quita columnas sin información |
| 3. Preprocesamiento | `src/preprocesamiento.py` | 5 clases, log(1+x) en columnas sesgadas, one-hot y normalización |
| 4. División | `src/preprocesamiento.py` | 80/20 estratificado (semilla 42); KDDTest+ como prueba |
| 5. Entrenamiento | `src/entrenar.py`, `src/modelos.py` | FT-Transformer (Keras) y XGBoost con pesos de clase |
| 6. Evaluación | `src/evaluacion.py` | Recall por clase, F1 macro, detección de ataques y falsas alarmas |
| 7. Rediseño v2 | `src/rediseno_umbral.py` | Alerta si P(normal) < umbral elegido con validación |
| 8. Registro | `resultados/registro_experimentos.csv` | Una fila por experimento |
| 9. Despliegue | *(entrega final)* | API con FastAPI en la nube |

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
python -m src.descargar_datos
python -m src.preprocesamiento
python -m src.entrenar --modelo xgboost
python -m src.entrenar --modelo transformer --epocas 15
python -m src.rediseno_umbral --modelo xgboost
python -m src.rediseno_umbral --modelo transformer
pytest -q
```

## Estructura
```
├── .github/workflows/ci.yml   # GitHub Actions: pruebas en cada cambio
├── data/
│   ├── raw/                   # datos originales (no se suben)
│   └── processed/             # datos listos para entrenar (no se suben)
├── docs/                      # informes de avance
├── models/                    # preprocesador y modelos entrenados (no se suben)
├── notebooks/                 # cuaderno de Colab (Avance III)
├── resultados/                # registro de experimentos y gráficas
├── src/
│   ├── config.py              # rutas, columnas y mapeo de ataques
│   ├── datos_modelo.py        # datos para cada modelo (tokens o one-hot)
│   ├── modelos.py             # arquitectura FT-Transformer
│   ├── entrenar.py            # entrenamiento y registro
│   ├── evaluacion.py          # métricas
│   ├── rediseno_umbral.py     # rediseño v2
│   ├── descargar_datos.py     # etapa 1
│   └── preprocesamiento.py    # etapas 2 a 4
├── tests/                     # pruebas del pipeline
├── requirements.txt
└── README.md
```

## Uso de IA
Se usó Claude (Anthropic) como apoyo para orientar el aprendizaje, estructurar el código y la documentación.
