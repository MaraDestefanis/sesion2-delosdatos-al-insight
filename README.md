# Sesión 2 — Del Dato al Insight

**Ciclo Magistral · Inteligencia de Datos: Análisis e IA Aplicada · 2026**  
Mara Destefanis · Mgter. Ciencia de Datos · Lic. Comunicación Social

---

## Dataset

**Avocado Prices** — Kaggle  
Precios y volúmenes semanales de paltas en el mercado estadounidense (2015–2023).  
Variables principales: `PrecioPromedio`, `TotalVolume`, tamaños de palta, tipo (conventional / organic), región geográfica.

---

## Notebooks

### 01 · Carga, transformación y exploración
Carga del dataset crudo, limpieza, transformación de tipos y exploración inicial.  
Perfilado de datos, detección de valores nulos y análisis de distribuciones.

### 02 · Descriptivo y diagnóstico
Análisis descriptivo por región, tipo y período.  
Diagnóstico de correlaciones, estacionalidad y comportamiento por tamaño de palta.

### 03 · Predictivo — LightGBM
Modelo de predicción de volumen semanal por región.  
- Estrategia: modelo global entrenado solo sobre paltas convencionales
- Features: calendario, lags (1, 4, 8, 52 semanas), rolling mean, PrecioPromedio
- Split temporal — últimas 12 semanas como test
- Optimización de hiperparámetros: `learning_rate=0.03`, `n_estimators=2000`
- **Error relativo final: 23.9%**

| Métrica | n=500 base | n=1000 | +Precio | lr=0.03 |
|---|---|---|---|---|
| MAE | 273,825 | 266,019 | 247,631 | 238,997 |
| RMSE | 577,878 | 574,252 | 572,571 | 555,500 |
| MAPE | 75.8% | 72.6% | 64.4% | 62.1% |
| Error relativo | 27.4% | 26.6% | 24.8% | 23.9% |

---

## Tablero
Dashboard interactivo construido con Streamlit.  
Visualización de métricas clave, evolución temporal y comparación por región.

---

## Próxima sesión
- Ajuste del modelo predictivo
- Integración del tablero con resultados del modelo
