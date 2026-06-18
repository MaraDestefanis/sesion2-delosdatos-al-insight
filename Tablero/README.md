# 🥑 De los Datos al Insight — Sesión 2
### Ciclo Magistral 2026 · Mara Destefanis

Repositorio de la segunda clase del Ciclo Magistral, dedicada al análisis completo
de un dataset real de ventas de paltas en el mercado estadounidense (2015–2023),
fuente: Hass Avocado Board (HAB).

---

## Estructura del proyecto

```
sesion2_Delosdatos-al-insight/
│
├── 01_carga_transf_exploracion.ipynb   # Carga, limpieza y EDA
├── 02_Descriptivo_Diagnostico.ipynb    # Análisis descriptivo y diagnóstico
│
├── Datos/
│   ├── 03-Predictivo.ipynb             # Modelo predictivo LightGBM
│   ├── palta_1.csv                     # Dataset procesado
│   └── metadata.ipynb                  # Exploración del metadata original
│
└── Tablero/
    ├── app.py                          # Dashboard interactivo Streamlit
    ├── requirements.txt                # Dependencias del entorno
    └── palta_1.csv                     # Dataset para el tablero
```

---

## ¿Qué se analiza?

El proyecto recorre el pipeline completo de un proyecto de datos:

**1. Carga y exploración** — transformación del dataset original, renombrado de columnas,
conversión de tipos, análisis de calidad y distribución de variables numéricas.

**2. Análisis descriptivo y diagnóstico** — visualizaciones interactivas con Plotly:
evolución semanal del volumen, participación por tamaño de palta (PLU 4046/4225/4770),
comparación convencional vs. orgánico, estacionalidad mensual y mapa de regiones.

**3. Modelo predictivo** — forecasting de demanda semanal por región con LightGBM,
usando lag features, promedios móviles y variables de calendario.
Proyección a 52 semanas por región con KPIs de variación vs. año anterior.

---

## Tablero interactivo

El tablero está desplegado en Streamlit Cloud:

👉 **[Ver tablero en vivo](#)** *(enlace disponible tras el deploy)*

Incluye tres secciones navegables:
- **Resumen ejecutivo** — KPIs + línea de tiempo + volumen por tamaño
- **Descriptivo y diagnóstico** — área apilada, estacionalidad, regiones y mapa
- **Proyección** — demanda proyectada 52 semanas por región con LightGBM

---

## Stack tecnológico

| Herramienta | Uso |
|-------------|-----|
| Python 3.13 | Lenguaje principal |
| pandas / numpy | Manipulación de datos |
| LightGBM | Modelo predictivo |
| Plotly | Visualizaciones interactivas |
| Streamlit | Dashboard web |
| scikit-learn | Métricas de evaluación |

---

## Cómo correr localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu_usuario/sesion2_Delosdatos-al-insight.git
cd sesion2_Delosdatos-al-insight/Tablero

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Correr el tablero
streamlit run app.py
```

---

## Sobre el dataset

- **Fuente**: Hass Avocado Board (HAB) vía Kaggle
- **Período**: 2015–2023
- **Registros**: ~18,000 filas originales
- **Variables clave**: precio promedio semanal (USD), volumen total (lbs),
  volumen por PLU (4046/4225/4770), bolsas, región y tipo (convencional/orgánico)

> **Nota**: el volumen está expresado en libras (lbs), no en unidades.
> Las columnas PLU representan el peso vendido de cada calibre de palta,
> no la cantidad de unidades. Las ventas en bolsa no permiten desagregar por PLU individual.

---

*Ciclo Magistral 2026 · [clases.maradestefanis.com](https://clases.maradestefanis.com)*
