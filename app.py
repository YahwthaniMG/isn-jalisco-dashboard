import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── Configuracion de pagina ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ISN Jalisco",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Fondo general */
.stApp {
    background-color: #0d0f14;
    color: #e8e3d5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #2a2d38;
}

/* Titulos */
h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    letter-spacing: -0.02em;
}

/* Tarjetas de metricas */
[data-testid="stMetric"] {
    background: #13161e;
    border: 1px solid #2a2d38;
    border-radius: 8px;
    padding: 16px 20px;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #e8e3d5;
}

/* Selectbox y multiselect */
[data-testid="stSelectbox"], [data-testid="stMultiSelect"] {
    font-family: 'DM Mono', monospace;
}

/* Alerta custom */
.alerta-box {
    background: #1a1208;
    border-left: 3px solid #f59e0b;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #fcd34d;
}

.alerta-roja {
    background: #1a0a0a;
    border-left: 3px solid #ef4444;
    color: #fca5a5;
}

.alerta-verde {
    background: #071a0e;
    border-left: 3px solid #22c55e;
    color: #86efac;
}

.alerta-gris {
    background: #111418;
    border-left: 3px solid #6b7280;
    color: #9ca3af;
}

.chip {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2a2d38;
    border-radius: 4px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #9ca3af;
    margin-right: 6px;
}

.titulo-seccion {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 4px;
}

.empresa-titulo {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e8e3d5;
    line-height: 1.1;
    margin-bottom: 4px;
}

.municipio-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #6366f1;
    letter-spacing: 0.05em;
}

div[data-testid="stHorizontalBlock"] {
    gap: 12px;
}
</style>
""", unsafe_allow_html=True)

ANOS = [2022, 2023, 2024, 2025, 2026]
ISN_TASA = 0.025  # Jalisco: 2.5%

# ── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("empresas_jalisco.csv")
    df.columns = ["Municipio", "Nombre", 2022, 2023, 2024, 2025, 2026]
    for a in ANOS:
        df[a] = pd.to_numeric(df[a], errors="coerce")
    df["primera_palabra"] = df["Nombre"].str.split().str[0].str.lower()
    df = df.sort_values("primera_palabra")
    return df

df = cargar_datos()

# ── Funciones de analisis ────────────────────────────────────────────────────
def calcular_nomina_estimada(impuesto):
    if pd.isna(impuesto):
        return None
    return impuesto / ISN_TASA

def calcular_salario_promedio(nomina, n_empleados):
    if nomina is None or n_empleados == 0:
        return None
    return nomina / 12 / n_empleados

def detectar_alertas(serie):
    alertas = []
    valores = [(a, v) for a, v in serie.items() if not pd.isna(v)]

    if len(valores) < 2:
        alertas.append(("gris", "Datos insuficientes para analisis de tendencia."))
        return alertas

    for i in range(1, len(valores)):
        a_prev, v_prev = valores[i - 1]
        a_curr, v_curr = valores[i]
        if v_prev == 0:
            continue
        cambio = (v_curr - v_prev) / v_prev * 100
        if cambio <= -40:
            alertas.append(("roja",
                f"{a_prev} -> {a_curr}: Caida de {cambio:.1f}%. "
                "Reduccion severa del ISN. Posible subdeclaracion de nomina, reduccion masiva de salarios o evasion fiscal. Se recomienda revision."))
        elif cambio <= -20:
            alertas.append(("amarilla",
                f"{a_prev} -> {a_curr}: Caida de {cambio:.1f}%. "
                "Reduccion significativa. Puede indicar recorte de personal (aunque se asume constante), renegociacion salarial o declaracion parcial."))
        elif cambio >= 50:
            alertas.append(("verde",
                f"{a_prev} -> {a_curr}: Incremento de {cambio:.1f}%. "
                "Crecimiento alto del ISN. Con empleados constantes, sugiere aumento salarial considerable o regularizacion de pagos anteriores."))
        elif cambio >= 20:
            alertas.append(("verde",
                f"{a_prev} -> {a_curr}: Incremento de {cambio:.1f}%. "
                "Crecimiento moderado-alto. Posible ajuste salarial por inflacion o prestaciones adicionales."))

    # Verificar años sin declarar
    anos_vacios = [str(a) for a, v in serie.items() if pd.isna(v)]
    if anos_vacios:
        alertas.append(("roja",
            f"Sin declaracion en: {', '.join(anos_vacios)}. "
            "Años sin registro de ISN. Puede indicar omision de declaracion o periodo sin actividad."))

    if not alertas:
        alertas.append(("gris", "Comportamiento estable. Sin variaciones significativas detectadas."))

    return alertas

def ranking_municipio(df, municipio, nombre_empresa):
    sub = df[df["Municipio"] == municipio].copy()
    sub["promedio"] = sub[ANOS].mean(axis=1)
    sub = sub.sort_values("promedio", ascending=False).reset_index(drop=True)
    pos = sub[sub["Nombre"] == nombre_empresa].index
    if len(pos) == 0:
        return None, len(sub)
    return pos[0] + 1, len(sub)

def promedio_municipal(df, municipio):
    sub = df[df["Municipio"] == municipio]
    return {a: sub[a].mean() for a in ANOS}

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p class='titulo-seccion'>Impuesto sobre Nomina</p>", unsafe_allow_html=True)
    st.markdown("## Jalisco ISN")
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.75rem;color:#4b5563;'>2022 – 2026</p>", unsafe_allow_html=True)
    st.divider()

    municipios = sorted(df["Municipio"].unique())
    municipio_sel = st.selectbox("Municipio", municipios, index=0)

    empresas_mun = df[df["Municipio"] == municipio_sel].sort_values("primera_palabra")
    lista_empresas = empresas_mun["Nombre"].tolist()

    busqueda = st.text_input("Buscar empresa", placeholder="Escribe un nombre...")
    if busqueda:
        lista_filtrada = [e for e in lista_empresas if busqueda.lower() in e.lower()]
    else:
        lista_filtrada = lista_empresas

    if not lista_filtrada:
        st.warning("Sin resultados para esa busqueda.")
        empresa_sel = lista_empresas[0] if lista_empresas else None
    else:
        empresa_sel = st.selectbox(
            f"Empresas en {municipio_sel} ({len(lista_filtrada)})",
            lista_filtrada
        )

    st.divider()

    n_empleados = st.number_input(
        "Numero de empleados (constante)",
        min_value=1, max_value=10000, value=50, step=1
    )

    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:#374151;margin-top:24px;'>Tasa ISN Jalisco: 2.5%</p>", unsafe_allow_html=True)

# ── CONTENIDO PRINCIPAL ──────────────────────────────────────────────────────
if empresa_sel is None:
    st.info("Selecciona un municipio y una empresa para ver el dashboard.")
    st.stop()

empresa_df = df[df["Nombre"] == empresa_sel].iloc[0]
serie = {a: empresa_df[a] for a in ANOS}

# Encabezado empresa
st.markdown(f"<p class='titulo-seccion'>Dashboard de empresa</p>", unsafe_allow_html=True)
st.markdown(f"<p class='empresa-titulo'>{empresa_sel}</p>", unsafe_allow_html=True)
st.markdown(f"<p class='municipio-tag'>{municipio_sel}</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────────────────────
valores_validos = {a: v for a, v in serie.items() if not pd.isna(v)}
ultimo_ano = max(valores_validos.keys()) if valores_validos else None
primer_ano = min(valores_validos.keys()) if valores_validos else None

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ultimo_val = valores_validos.get(ultimo_ano, None)
    st.metric(
        label=f"ISN {ultimo_ano}",
        value=f"${ultimo_val:,.0f}" if ultimo_val else "N/D"
    )

with col2:
    nomina_est = calcular_nomina_estimada(ultimo_val)
    st.metric(
        label="Nomina estimada",
        value=f"${nomina_est:,.0f}" if nomina_est else "N/D"
    )

with col3:
    sal = calcular_salario_promedio(nomina_est, n_empleados)
    st.metric(
        label="Salario mens. prom.",
        value=f"${sal:,.0f}" if sal else "N/D"
    )

with col4:
    if ultimo_ano and primer_ano and ultimo_ano != primer_ano:
        v_ini = valores_validos.get(primer_ano)
        v_fin = valores_validos.get(ultimo_ano)
        if v_ini and v_fin:
            crecimiento = (v_fin - v_ini) / v_ini * 100
            st.metric(
                label=f"Crecimiento {primer_ano}-{ultimo_ano}",
                value=f"{crecimiento:+.1f}%"
            )
        else:
            st.metric(label="Crecimiento total", value="N/D")
    else:
        st.metric(label="Crecimiento total", value="N/D")

with col5:
    rank, total = ranking_municipio(df, municipio_sel, empresa_sel)
    st.metric(
        label=f"Ranking en {municipio_sel}",
        value=f"#{rank} / {total}" if rank else "N/D"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── GRAFICAS ─────────────────────────────────────────────────────────────────
prom_mun = promedio_municipal(df, municipio_sel)

col_graf1, col_graf2 = st.columns([3, 2])

with col_graf1:
    st.markdown("<p class='titulo-seccion'>Evolucion ISN declarado</p>", unsafe_allow_html=True)

    anos_str = [str(a) for a in ANOS]
    vals_empresa = [serie[a] for a in ANOS]
    vals_promedio = [prom_mun[a] for a in ANOS]

    fig = go.Figure()

    # Relleno entre curvas
    fig.add_trace(go.Scatter(
        x=anos_str + anos_str[::-1],
        y=vals_empresa + vals_promedio[::-1],
        fill='toself',
        fillcolor='rgba(99,102,241,0.06)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Promedio municipal
    fig.add_trace(go.Scatter(
        x=anos_str,
        y=vals_promedio,
        name=f"Prom. {municipio_sel}",
        mode="lines",
        line=dict(color="#374151", width=1.5, dash="dot"),
        hovertemplate="Promedio: $%{y:,.0f}<extra></extra>"
    ))

    # Empresa
    fig.add_trace(go.Scatter(
        x=anos_str,
        y=vals_empresa,
        name=empresa_sel,
        mode="lines+markers",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(
            size=8,
            color=["#6366f1" if not pd.isna(v) else "#1e2130" for v in vals_empresa],
            line=dict(color="#6366f1", width=2)
        ),
        hovertemplate="ISN: $%{y:,.0f}<extra></extra>",
        connectgaps=False
    ))

    fig.update_layout(
        plot_bgcolor="#0d0f14",
        paper_bgcolor="#0d0f14",
        font=dict(family="DM Mono", color="#6b7280", size=11),
        legend=dict(
            bgcolor="#13161e",
            bordercolor="#2a2d38",
            borderwidth=1,
            font=dict(size=10)
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor="#2a2d38",
            tickcolor="#2a2d38"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1a1d27",
            showline=False,
            tickprefix="$",
            tickformat=",.0f"
        ),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=0),
        height=320
    )

    st.plotly_chart(fig, use_container_width=True)

with col_graf2:
    st.markdown("<p class='titulo-seccion'>Nomina estimada por año</p>", unsafe_allow_html=True)

    nominas = [v / ISN_TASA if not pd.isna(v) else None for v in vals_empresa]
    colores_barras = ["#6366f1" if n is not None else "#1e2130" for n in nominas]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=anos_str,
        y=nominas,
        marker_color=colores_barras,
        marker_line_width=0,
        hovertemplate="Nomina: $%{y:,.0f}<extra></extra>"
    ))

    fig2.update_layout(
        plot_bgcolor="#0d0f14",
        paper_bgcolor="#0d0f14",
        font=dict(family="DM Mono", color="#6b7280", size=11),
        xaxis=dict(showgrid=False, showline=True, linecolor="#2a2d38"),
        yaxis=dict(showgrid=True, gridcolor="#1a1d27", tickprefix="$", tickformat=",.0f"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=320,
        showlegend=False
    )

    st.plotly_chart(fig2, use_container_width=True)

# ── ALERTAS ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p class='titulo-seccion'>Analisis de riesgo fiscal</p>", unsafe_allow_html=True)
st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:0.75rem;color:#4b5563;margin-bottom:12px;'>Con {n_empleados} empleados constantes, las variaciones en ISN reflejan cambios en salarios o irregularidades.</p>", unsafe_allow_html=True)

alertas = detectar_alertas(serie)
for tipo, texto in alertas:
    clase = {
        "roja": "alerta-box alerta-roja",
        "amarilla": "alerta-box",
        "verde": "alerta-box alerta-verde",
        "gris": "alerta-box alerta-gris"
    }.get(tipo, "alerta-box alerta-gris")
    icono = {"roja": "!", "amarilla": "~", "verde": "+", "gris": "i"}.get(tipo, "i")
    st.markdown(f'<div class="{clase}">[{icono}] {texto}</div>', unsafe_allow_html=True)

# ── TABLA COMPARATIVA CON MUNICIPIO ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p class='titulo-seccion'>Comparativa vs promedio municipal</p>", unsafe_allow_html=True)

tabla_data = []
for a in ANOS:
    v_emp = serie[a]
    v_prom = prom_mun[a]
    if not pd.isna(v_emp) and not pd.isna(v_prom) and v_prom > 0:
        diff = (v_emp - v_prom) / v_prom * 100
        diff_str = f"{diff:+.1f}%"
    else:
        diff_str = "N/D"
    tabla_data.append({
        "Año": str(a),
        "ISN empresa": f"${v_emp:,.0f}" if not pd.isna(v_emp) else "Sin declarar",
        "Promedio municipal": f"${v_prom:,.2f}" if not pd.isna(v_prom) else "N/D",
        "Diferencia vs prom.": diff_str
    })

tabla_df = pd.DataFrame(tabla_data)
st.dataframe(
    tabla_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Año": st.column_config.TextColumn(width="small"),
        "Diferencia vs prom.": st.column_config.TextColumn(width="medium"),
    }
)

# ── RESUMEN MUNICIPAL ─────────────────────────────────────────────────────────
with st.expander(f"Resumen general del municipio: {municipio_sel}"):
    empresas_mun_df = df[df["Municipio"] == municipio_sel]
    total_empresas = len(empresas_mun_df)
    total_recaudado = empresas_mun_df[ANOS].sum().sum()
    empresa_top = empresas_mun_df.copy()
    empresa_top["prom"] = empresa_top[ANOS].mean(axis=1)
    empresa_top = empresa_top.sort_values("prom", ascending=False).iloc[0]

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Empresas registradas", total_empresas)
    with mc2:
        st.metric("ISN total acumulado", f"${total_recaudado:,.0f}")
    with mc3:
        st.metric("Mayor contribuyente", empresa_top["Nombre"][:28] + "..." if len(empresa_top["Nombre"]) > 28 else empresa_top["Nombre"])
