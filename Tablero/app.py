# ── app.py — Tablero Paltas · Streamlit ───────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy  as np
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express       as px
import plotly.io            as pio
import lightgbm             as lgb

# ── Paleta ────────────────────────────────────────────────────────────────────
PALETA = {
    "hueso":   "#F2EFE4",
    "oliva":   "#7C8C5E",
    "tierra":  "#A0785A",
    "musgo":   "#4F6347",
    "pizarra": "#5A6472",
    "crema":   "#D9D2C0",
    "ink":     "#2C2C2C",
}
COLORES_SEQ = [PALETA["oliva"], PALETA["tierra"],
               PALETA["pizarra"], PALETA["musgo"], PALETA["crema"]]

TEMPLATE_PALTA = go.layout.Template(
    layout=go.Layout(
        font          = dict(family="Georgia, serif", color=PALETA["ink"], size=12),
        paper_bgcolor = "white",
        plot_bgcolor  = "white",
        title         = dict(font=dict(size=14, color=PALETA["ink"]), x=0.5, xanchor="center"),
        xaxis         = dict(showgrid=False, linecolor=PALETA["ink"], linewidth=0.7),
        yaxis         = dict(gridcolor=PALETA["crema"], gridwidth=0.5,
                             linecolor=PALETA["ink"], linewidth=0.7),
        legend        = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=10)),
        colorway      = COLORES_SEQ,
    )
)
pio.templates["palta"] = TEMPLATE_PALTA
pio.templates.default  = "palta"

# ── Config página ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Tablero Paltas · HAB",
    page_icon  = "🥑",
    layout     = "wide",
)

# ── CSS mínimo ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .main {{ background-color: {PALETA['hueso']}; }}
    .block-container {{ padding-top: 1.5rem; }}
    .kpi-card {{
        background-color: white;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 4px solid {PALETA['oliva']};
        margin-bottom: 0.5rem;
    }}
    .kpi-label {{ font-size: 0.78rem; color: {PALETA['pizarra']}; margin-bottom: 2px; }}
    .kpi-value {{ font-size: 1.4rem; font-weight: 600; color: {PALETA['ink']}; }}
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv(
        r"C:\Users\ASUS\Documents\Teacher\Clases_Magistrales\sesion2_Delosdatos-al-insight\Tablero\palta_1.csv",
        parse_dates=["Fecha"]
    )
    return df

@st.cache_resource
def entrenar_modelo(df):
    data = df[["Fecha", "Region_Geo", "Tipo", "TotalVolume"]].copy()
    data = data.sort_values(["Region_Geo", "Tipo", "Fecha"]).reset_index(drop=True)

    data["anio"]      = data["Fecha"].dt.year
    data["mes"]       = data["Fecha"].dt.month
    data["semana"]    = data["Fecha"].dt.isocalendar().week.astype(int)
    data["trimestre"] = data["Fecha"].dt.quarter

    grp = data.groupby(["Region_Geo", "Tipo"])["TotalVolume"]
    data["lag_1"]  = grp.shift(1)
    data["lag_4"]  = grp.shift(4)
    data["lag_8"]  = grp.shift(8)
    data["lag_52"] = grp.shift(52)
    data["roll_4"] = grp.shift(1).transform(lambda x: x.rolling(4,  min_periods=1).mean())
    data["roll_8"] = grp.shift(1).transform(lambda x: x.rolling(8,  min_periods=1).mean())

    data["Region_Geo"] = data["Region_Geo"].astype("category")
    data["Tipo"]       = data["Tipo"].astype("category")
    data = data.dropna(subset=["lag_1", "lag_4", "lag_8"]).reset_index(drop=True)

    FEATURES = ["anio", "mes", "semana", "trimestre",
                "lag_1", "lag_4", "lag_8", "lag_52",
                "roll_4", "roll_8", "Region_Geo", "Tipo"]

    fecha_corte = data["Fecha"].max() - pd.Timedelta(weeks=12)
    train = data[data["Fecha"] <= fecha_corte]

    modelo = lgb.LGBMRegressor(
        objective="regression", learning_rate=0.05,
        num_leaves=31, n_estimators=500, verbose=-1
    )
    modelo.fit(
        train[FEATURES], train["TotalVolume"],
        categorical_feature=["Region_Geo", "Tipo"]
    )
    return modelo, data, FEATURES

palta_1 = cargar_datos()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🥑 Tablero Paltas")
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

fecha_min = palta_1["Fecha"].min().date()
fecha_max = palta_1["Fecha"].max().date()
rango = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max],
                               min_value=fecha_min, max_value=fecha_max)

regiones   = ["Todas"] + sorted(palta_1["Region_Geo"].dropna().unique().tolist())
region_sel = st.sidebar.selectbox("Región", regiones)

tipos      = ["Todos", "conventional", "organic"]
tipo_sel   = st.sidebar.selectbox("Tipo", tipos)

# Filtro aplicado
df_f = palta_1.copy()
if len(rango) == 2:
    df_f = df_f[(df_f["Fecha"] >= pd.Timestamp(rango[0])) &
                (df_f["Fecha"] <= pd.Timestamp(rango[1]))]
if region_sel != "Todas":
    df_f = df_f[df_f["Region_Geo"] == region_sel]
if tipo_sel != "Todos":
    df_f = df_f[df_f["Tipo"] == tipo_sel]

st.sidebar.markdown("---")
pagina = st.sidebar.radio("Sección", ["Resumen ejecutivo", "Descriptivo y diagnóstico", "Proyección"])

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ═════════════════════════════════════════════════════════════════════════════
if pagina == "Resumen ejecutivo":
    st.title("🥑 Mercado de Paltas · HAB Dataset")
    st.markdown("---")

    vol_total    = df_f["TotalVolume"].sum()
    vol_semanal  = df_f.groupby("Fecha")["TotalVolume"].sum().mean()
    region_lider = df_f.groupby("Region_Geo")["TotalVolume"].sum().idxmax() if region_sel == "Todas" else region_sel
    tipo_dom     = df_f.groupby("Tipo")["TotalVolume"].sum().idxmax() if tipo_sel == "Todos" else tipo_sel
    semana_pico  = df_f.groupby("Fecha")["TotalVolume"].sum().idxmax().strftime("%d %b %Y")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in zip(
        [c1, c2, c3, c4, c5],
        ["Volumen total (lbs)", "Promedio semanal (lbs)", "Región líder", "Tipo dominante", "Semana pico"],
        [f"{vol_total:,.0f}", f"{vol_semanal:,.0f}", region_lider, tipo_dom.capitalize(), semana_pico]
    ):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # G1 — Línea de tiempo
    vol_sem = df_f.groupby("Fecha")["TotalVolume"].sum().reset_index()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=vol_sem["Fecha"], y=vol_sem["TotalVolume"],
        mode="lines", name="Volumen total",
        line=dict(color=PALETA["oliva"], width=1.8),
        fill="tozeroy", fillcolor="rgba(124,140,94,0.12)",
    ))
    fig1.update_layout(title="Volumen semanal de paltas vendidas (lbs)",
                       yaxis=dict(tickformat=",.0f"), height=380)
    st.plotly_chart(fig1, use_container_width=True)

    # G2 — Barras por tamaño
    vol_tam = pd.DataFrame({
        "Tamaño": ["Normal (PLU 4046)", "Grande (PLU 4225)", "Extra grande (PLU 4770)"],
        "Volumen": [df_f["Palta_Normal"].sum(),
                    df_f["Palta_Grande"].sum(),
                    df_f["Palta_Extragrande"].sum()]
    }).sort_values("Volumen")

    fig2 = go.Figure(go.Bar(
        x=vol_tam["Volumen"], y=vol_tam["Tamaño"], orientation="h",
        marker_color=[PALETA["musgo"], PALETA["tierra"], PALETA["oliva"]],
        text=vol_tam["Volumen"].apply(lambda x: f"{x:,.0f}"),
        textposition="outside",
    ))
    fig2.update_layout(title="Volumen total por tamaño de palta (lbs)",
                       xaxis=dict(tickformat=",.0f"), height=300,
                       margin=dict(l=160, r=120, t=50, b=40))
    st.plotly_chart(fig2, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — DESCRIPTIVO Y DIAGNÓSTICO
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "Descriptivo y diagnóstico":
    st.title("Análisis descriptivo y diagnóstico")
    st.markdown("---")

    # G3 — Área apilada
    vol_tipo = df_f.groupby(["Fecha", "Tipo"])["TotalVolume"].sum().reset_index()
    fig3 = go.Figure()
    for tipo, color in [("conventional", PALETA["oliva"]), ("organic", PALETA["tierra"])]:
        d = vol_tipo[vol_tipo["Tipo"] == tipo]
        fig3.add_trace(go.Scatter(
            x=d["Fecha"], y=d["TotalVolume"],
            name=tipo.capitalize(), mode="lines",
            stackgroup="one",
            line=dict(color=color, width=0.8),
        ))
    fig3.update_layout(title="Volumen semanal por tipo (lbs)",
                       yaxis=dict(tickformat=",.0f"), height=400,
                       legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    st.plotly_chart(fig3, use_container_width=True)

    # G4 — Estacionalidad
    MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    est = df_f.groupby([df_f["Fecha"].dt.month, "Tipo"])["TotalVolume"].mean().reset_index()
    est.columns = ["Mes", "Tipo", "TotalVolume"]
    est["Mes_nombre"] = est["Mes"].apply(lambda x: MESES[x-1])

    fig4 = go.Figure()
    for tipo, color in [("conventional", PALETA["oliva"]), ("organic", PALETA["tierra"])]:
        d = est[est["Tipo"] == tipo]
        fig4.add_trace(go.Scatter(
            x=d["Mes_nombre"], y=d["TotalVolume"],
            name=tipo.capitalize(), mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=7, color=color),
        ))
    fig4.update_layout(title="Estacionalidad — volumen promedio por mes",
                       yaxis=dict(tickformat=",.0f"), height=400,
                       xaxis=dict(categoryorder="array", categoryarray=MESES),
                       legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))
    st.plotly_chart(fig4, use_container_width=True)

    col1, col2 = st.columns(2)

    # G5 — Regiones
    with col1:
        vol_reg = df_f.groupby(["Region_Geo", "Tipo"])["TotalVolume"].sum().reset_index()
        orden   = vol_reg.groupby("Region_Geo")["TotalVolume"].sum().sort_values().index
        fig5 = go.Figure()
        for tipo, color in [("conventional", PALETA["oliva"]), ("organic", PALETA["tierra"])]:
            d = vol_reg[vol_reg["Tipo"] == tipo]
            fig5.add_trace(go.Bar(
                x=d["TotalVolume"], y=d["Region_Geo"], orientation="h",
                name=tipo.capitalize(), marker_color=color,
            ))
        fig5.update_layout(
            title="Volumen por región (lbs)", barmode="group",
            yaxis=dict(categoryorder="array", categoryarray=list(orden), tickfont=dict(size=10)),
            xaxis=dict(tickformat=",.0f"), height=500,
            margin=dict(l=130, r=80, t=60, b=40),
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # G6 — Mapa
    with col2:
        COORDS = {
            "California":(36.77,-119.41),"SouthCentral":(31.96,-99.90),
            "Northeast":(42.40,-71.38),"Southeast":(32.16,-82.90),
            "Midsouth":(35.51,-86.58),"GreatLakes":(43.69,-84.95),
            "Plains":(41.12,-100.26),"West":(40.58,-111.09),
            "NewYork":(40.71,-74.00),"LosAngeles":(34.05,-118.24),
            "Chicago":(41.87,-87.62),"Houston":(29.76,-95.36),
            "Atlanta":(33.74,-84.38),"DallasFtWorth":(32.77,-96.79),
            "Seattle":(47.60,-122.33),"Portland":(45.50,-122.67),
            "Denver":(39.73,-104.99),"PhoenixTucson":(33.44,-112.07),
            "SanFrancisco":(37.77,-122.41),"Sacramento":(38.58,-121.49),
            "LasVegas":(36.16,-115.13),"SanDiego":(32.71,-117.16),
            "Nashville":(36.16,-86.78),"Tampa":(27.95,-82.45),
            "Miami":(25.76,-80.19),"Boston":(42.36,-71.05),
            "Philadelphia":(39.95,-75.16),"Charlotte":(35.22,-80.84),
            "RaleighGreensboro":(35.77,-78.63),"Louisville":(38.25,-85.75),
            "Indianapolis":(39.76,-86.15),"Columbus":(39.96,-82.99),
            "Cincinnati":(39.10,-84.51),"Detroit":(42.33,-83.04),
            "StLouis":(38.62,-90.19),"KansasCity":(39.09,-94.57),
            "Minneapolis":(44.97,-93.26),"Milwaukee":(43.03,-87.90),
            "Omaha":(41.25,-95.93),"Albany":(42.65,-73.75),
            "BuffaloRochester":(42.88,-78.87),"Pittsburgh":(40.44,-79.99),
            "Hartford":(41.76,-72.68),"Richmond":(37.54,-77.43),
            "Jacksonville":(30.33,-81.65),"Orlando":(28.53,-81.37),
            "NewOrleans":(29.95,-90.07),"Spokane":(47.65,-117.42),
            "Boise":(43.61,-116.20),"NorthernNewEngland":(44.31,-72.99),
            "WestTexCapRock":(31.54,-102.57),"GrandRapids":(42.96,-85.66),
            "SouthCarolina":(33.83,-81.16),"Tennessee":(35.86,-86.66),
        }
        vm = df_f.groupby("Region_Geo")["TotalVolume"].sum().reset_index()
        vm["lat"] = vm["Region_Geo"].map(lambda r: COORDS.get(r,(None,None))[0])
        vm["lon"] = vm["Region_Geo"].map(lambda r: COORDS.get(r,(None,None))[1])
        vm = vm.dropna(subset=["lat","lon"])

        fig6 = px.scatter_geo(
            vm, lat="lat", lon="lon", size="TotalVolume",
            hover_name="Region_Geo",
            hover_data={"TotalVolume":":,.0f","lat":False,"lon":False},
            color="TotalVolume",
            color_continuous_scale=[PALETA["crema"], PALETA["oliva"], PALETA["musgo"]],
            scope="usa", title="Volumen por región (mapa)", size_max=50,
        )
        fig6.update_layout(
            height=500,
            geo=dict(bgcolor="white", lakecolor=PALETA["hueso"],
                     landcolor=PALETA["hueso"], showlakes=True, showland=True),
            coloraxis_colorbar=dict(title="Lbs", tickformat=",.0f", len=0.6),
        )
        st.plotly_chart(fig6, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — PROYECCIÓN
# ═════════════════════════════════════════════════════════════════════════════
elif pagina == "Proyección":
    st.title("Proyección — próximas 52 semanas")
    st.markdown("---")

    with st.spinner("Entrenando modelo..."):
        modelo, df_model, FEATURES = entrenar_modelo(palta_1)

    col1, col2 = st.columns(2)
    with col1:
        reg_proy = st.selectbox("Región", ["Nacional (agregado)"] +
                                 sorted(palta_1["Region_Geo"].dropna().unique().tolist()))
    with col2:
        tipo_proy = st.selectbox("Tipo", ["Todos", "conventional", "organic"])

    # Histórico filtrado
    hist = palta_1.copy()
    if reg_proy != "Nacional (agregado)":
        hist = hist[hist["Region_Geo"] == reg_proy]
    if tipo_proy != "Todos":
        hist = hist[hist["Tipo"] == tipo_proy]
    hist_sem = hist.groupby("Fecha")["TotalVolume"].sum().reset_index().sort_values("Fecha")

    # Construcción df_futuro
    ultima_fecha  = palta_1["Fecha"].max()
    fechas_fut    = [ultima_fecha + pd.Timedelta(weeks=i) for i in range(1, 53)]

    regiones_proy = palta_1["Region_Geo"].dropna().unique() if reg_proy == "Nacional (agregado)" else [reg_proy]
    tipos_proy    = ["conventional", "organic"] if tipo_proy == "Todos" else [tipo_proy]

    hist_lags = (
        palta_1
        .groupby(["Region_Geo", "Tipo", "Fecha"])["TotalVolume"]
        .sum()
        .reset_index()
        .sort_values(["Region_Geo", "Tipo", "Fecha"])
    )

    registros_fut = []
    for region in regiones_proy:
        for tipo in tipos_proy:
            serie = (
                hist_lags[(hist_lags["Region_Geo"] == region) & (hist_lags["Tipo"] == tipo)]
                ["TotalVolume"].tolist()
            )
            for fecha in fechas_fut:
                lag_1  = serie[-1]  if len(serie) >= 1  else 0
                lag_4  = serie[-4]  if len(serie) >= 4  else 0
                lag_8  = serie[-8]  if len(serie) >= 8  else 0
                lag_52 = serie[-52] if len(serie) >= 52 else 0
                roll_4 = np.mean(serie[-4:]) if len(serie) >= 4 else np.mean(serie)
                roll_8 = np.mean(serie[-8:]) if len(serie) >= 8 else np.mean(serie)

                row = {
                    "Fecha": fecha, "Region_Geo": region, "Tipo": tipo,
                    "anio": fecha.year, "mes": fecha.month,
                    "semana": fecha.isocalendar()[1], "trimestre": (fecha.month-1)//3+1,
                    "lag_1": lag_1, "lag_4": lag_4, "lag_8": lag_8, "lag_52": lag_52,
                    "roll_4": roll_4, "roll_8": roll_8,
                }
                registros_fut.append(row)

                pred = modelo.predict(
                    pd.DataFrame([row])[FEATURES]
                    .assign(Region_Geo=lambda d: d["Region_Geo"].astype("category"),
                            Tipo=lambda d: d["Tipo"].astype("category"))
                )[0]
                serie.append(max(pred, 0))

    df_fut = pd.DataFrame(registros_fut)
    df_fut["Region_Geo"] = df_fut["Region_Geo"].astype("category")
    df_fut["Tipo"]       = df_fut["Tipo"].astype("category")
    df_fut["Predicho"]   = modelo.predict(df_fut[FEATURES]).clip(min=0)

    proj_sem = df_fut.groupby("Fecha")["Predicho"].sum().reset_index()

    # KPIs proyección
    vol_proy_total = proj_sem["Predicho"].sum()
    vol_hist_año   = hist_sem[hist_sem["Fecha"] >= (ultima_fecha - pd.Timedelta(weeks=52))]["TotalVolume"].sum()
    variacion      = (vol_proy_total - vol_hist_año) / vol_hist_año * 100 if vol_hist_año > 0 else 0
    semana_pico_fut = proj_sem.loc[proj_sem["Predicho"].idxmax(), "Fecha"].strftime("%d %b %Y")
    vol_pico_fut    = proj_sem["Predicho"].max()

    k1, k2, k3 = st.columns(3)
    for col, label, value in zip(
        [k1, k2, k3],
        ["Volumen proyectado (52 sem)", "Variación vs año anterior", "Semana pico proyectada"],
        [f"{vol_proy_total:,.0f} lbs",
         f"{'▲' if variacion > 0 else '▼'} {abs(variacion):.1f}%",
         f"{semana_pico_fut} · {vol_pico_fut:,.0f} lbs"]
    ):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráfico histórico + proyección
    fig_proy = go.Figure()

    hist_rec = hist_sem.tail(52)
    fig_proy.add_trace(go.Scatter(
        x=hist_rec["Fecha"], y=hist_rec["TotalVolume"],
        mode="lines", name="Histórico",
        line=dict(color=PALETA["musgo"], width=2),
    ))
    fig_proy.add_trace(go.Scatter(
        x=proj_sem["Fecha"], y=proj_sem["Predicho"],
        mode="lines", name="Proyección",
        line=dict(color=PALETA["tierra"], width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(160,120,90,0.08)",
    ))
    fig_proy.add_vline(
        x=ultima_fecha, line_dash="dot",
        line_color=PALETA["pizarra"], line_width=1.2,
        annotation_text="Hoy", annotation_position="top right",
    )
    fig_proy.update_layout(
        title       = f"Proyección de demanda — próximas 52 semanas · {reg_proy}",
        xaxis_title = "Semana",
        yaxis_title = "Libras vendidas",
        yaxis       = dict(tickformat=",.0f"),
        height      = 460,
        legend      = dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_proy, use_container_width=True)
