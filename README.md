# INF8239_U01 — Clasificación de calidad del aire con SVM

Proyecto de Ciencia de Datos II (Maestría en Ciencia de Datos e Inteligencia
Artificial). Entrena y evalúa una **Máquina de Vectores de Soporte (SVM)** para
clasificar el nivel de calidad del aire de una región (Good, Moderate, Poor,
Hazardous) a partir de variables ambientales y demográficas.

**Dataset:** *Air Quality and Pollution Assessment* (Kaggle · Mujtaba Mateen ·
**Apache 2.0**), 5,000 muestras y 9 predictores numéricos. Uso académico aprobado.

---

## Estructura del proyecto

```
INF8239_U01/
├── data/raw/
│   └── dataset.csv             # dataset descargado 
├── docs/
│   └── ficha_dataset.md        # ficha de procedencia, target y licencia
├── images/
│   └── confusion_matrix.png    # gráficos generados
├── notebooks/
│   ├── 00_verificacion.ipynb   # verificación del entorno
│   ├── 01_svm_guiada.ipynb     # LAB01 · SVM guiado (breast cancer)
│   └── 02_svm_autorizado.ipynb # LAB02 · SVM con dataset aprobado (calidad del aire)
├── reports/
│   ├── svm_best.joblib         # mejor modelo serializado
│   └── svm_cv_results.csv      # resultados de validación cruzada
├── src/inf8239_u01/
│   ├── __init__.py
│   ├── data.py                 # descarga reproducible (Kaggle API)
│   ├── environment.py
│   └── models.py
├── tests/
│   ├── test_data_contract.py   # contrato de datos del LAB02 (3 pruebas)
│   ├── test_environment.py
│   └── test_models.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Requisitos

- **Python 3.10+**
- Dependencias en `requirements.txt` (`pandas`, `scikit-learn`, `matplotlib`,
  `joblib`, `kagglehub`, `jupyter`/`ipykernel`, `pytest`).
- **Credenciales de Kaggle** (para descargar el dataset).

---

## Instalación

```powershell
# Desde la raíz del proyecto
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Credenciales de Kaggle (una sola vez)

1. En kaggle.com → tu avatar → **Settings** → **API** → **Create New Token**.
2. Coloca el `kaggle.json` descargado en `C:\Users\<usuario>\.kaggle\kaggle.json`
   (o define las variables `KAGGLE_USERNAME` y `KAGGLE_KEY`).

---

## Descarga del dataset

La descarga está encapsulada en `src/inf8239_u01/data.py` (la fuente requiere
autenticación, no expone URL `.csv` directa). Desde el notebook:

```python
from src.inf8239_u01.data import download_csv
URL = "https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment"
path = download_csv(URL)   # guarda data/raw/dataset.csv
```

---

## Target y métrica

- **Target:** `Air Quality` — clasificación **multiclase** (Good, Moderate, Poor,
  Hazardous).
- **Métrica principal:** **F1 macro** (da igual peso a las 4 clases; adecuada ante el
  desbalance del dataset).
- **Error más costoso:** clasificar como "Good/Moderate" una región "Hazardous"
  (no se emite alerta y la población sensible queda expuesta).

---

## Cómo reproducir

1. Activa el entorno virtual e instala dependencias.
2. Configura las credenciales de Kaggle.
3. Selecciona el kernel `.venv` en el notebook.
4. Ejecuta `notebooks/01_svm_guiada.ipynb` en orden (pasos 1–11).

El notebook es **determinista** (`random_state=42` en división y modelo).

---

## Pruebas

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
```

Verifican el contrato de datos: dataset no vacío, columnas clave presentes y target
sin nulos con ≥2 clases (3 pruebas).

---

## Resultados

Distribución de clases (desbalanceado): Good 2000 (40%), Moderate 1500 (30%),
Poor 1000 (20%), Hazardous 500 (10%).

| Modelo | F1 macro (prueba) |
|--------|-------------------|
| Baseline (DummyClassifier) | 0.1429 |
| SVM (RBF, C=1) | 0.9127 |

Reporte por clase del SVM (conjunto de prueba):

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Good | 0.9925 | 0.9950 | 0.9938 |
| Moderate | 0.9379 | 0.9567 | 0.9472 |
| Poor | 0.8600 | 0.8600 | 0.8600 |
| Hazardous | 0.8817 | 0.8200 | 0.8497 |

Accuracy global: **0.9390**. El SVM supera ampliamente la línea base. La clase de
mayor riesgo, **Hazardous, es la más difícil (recall 0.82)**: el modelo deja pasar
~18% de las regiones realmente peligrosas, que es el error más costoso a vigilar.

---

## Diccionario de datos

| Variable | Significado | Unidad | Disponible al predecir | Transformación | Riesgo |
|----------|-------------|--------|------------------------|----------------|--------|
| Temperature | Temperatura media de la región | °C | Sí | StandardScaler | Outliers estacionales |
| Humidity | Humedad relativa | % | Sí | StandardScaler | Valores fuera de [0–100] |
| PM2.5 | Partículas finas | µg/m³ | Sí | StandardScaler | Correlación con PM10 |
| PM10 | Partículas gruesas | µg/m³ | Sí | StandardScaler | Correlación con PM2.5 |
| NO2 | Dióxido de nitrógeno | ppb | Sí | StandardScaler | Outliers industriales |
| SO2 | Dióxido de azufre | ppb | Sí | StandardScaler | Outliers industriales |
| CO | Monóxido de carbono | ppm | Sí | StandardScaler | Escala distinta |
| Proximity_to_Industrial_Areas | Distancia a zona industrial | km | Sí | StandardScaler | Proxy de contaminación |
| Population_Density | Densidad poblacional | hab/km² | Sí | StandardScaler | Sesgo urbano/rural |
| Air Quality | Nivel de calidad del aire (**target**) | categoría | — | Etiqueta de clase | Clases desbalanceadas |

---

## Licencia del dataset

*Air Quality and Pollution Assessment* — **Apache 2.0**. Uso académico.

---

## Conclusión

Este trabajo abordó la clasificación del nivel de calidad del aire de una región
—Good, Moderate, Poor o Hazardous— a partir de nueve variables ambientales y
demográficas, usando el dataset *Air Quality and Pollution Assessment* de Kaggle
(licencia Apache 2.0, uso académico aprobado). La unidad de análisis es una región
caracterizada por sus mediciones, y la decisión que apoya el modelo es la emisión de
alertas de contaminación.

La selección siguió criterios explícitos: licencia permisiva, 5,000 filas, target
observable con cuatro clases, variables disponibles al predecir y tamaño compatible con
CPU. Se compararon dos candidatos y se eligió este por ser 100% numérico, directamente
compatible con SVM y StandardScaler sin codificación categórica.

La auditoría confirmó un conjunto limpio: sin duplicados, sin valores faltantes y con
las nueve predictoras numéricas. Al no existir identificadores ni fugas, no se eliminó
ninguna columna, respetando el criterio de que toda exclusión debe tener una razón
semántica o de disponibilidad. El preprocesamiento (imputación por mediana y
estandarización) se encapsuló en un ColumnTransformer dentro de un Pipeline, de modo que
el escalado se ajusta solo con los datos de entrenamiento, evitando el data leakage.

Se dividió el conjunto en 80/20 de forma estratificada y se comparó un baseline
(DummyClassifier) contra un SVM con kernel RBF, evaluando con F1 macro para dar igual
peso a las cuatro clases. Los resultados son claros: el baseline obtuvo un F1 macro de
0.1429 —al predecir siempre la clase mayoritaria falla en las demás—, mientras que el
SVM alcanzó 0.9127, con una exactitud global de 0.9390. Esto confirma que las variables
ambientales contienen señal predictiva fuerte.

El análisis por clase es el hallazgo más relevante. El dataset está desbalanceado (Good
40%, Moderate 30%, Poor 20%, Hazardous 10%), y precisamente la clase Hazardous —la de
mayor interés— es la más difícil, con un recall de 0.82: el modelo deja pasar cerca del
18% de las regiones realmente peligrosas. Como el error más costoso, definido en la
ficha, es no alertar ante aire peligroso, este es el punto crítico a mejorar; una
extensión natural sería aplicar class_weight="balanced" o ajustar el umbral de decisión
para elevar el recall de las clases de riesgo.

La reproducibilidad se aseguró con descarga encapsulada vía API de Kaggle, rutas
resueltas contra la raíz del proyecto, semilla fija y un contrato de datos con tres
pruebas automatizadas. Como limitación, el dataset es sintético, por lo que un despliegue
real exigiría validar con mediciones auténticas y vigilar el sesgo territorial. En
conjunto, el proyecto demuestra un flujo completo y trazable de selección, auditoría y
modelado supervisado sobre datos propios.