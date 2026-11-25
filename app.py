# ==============================================================================
# DASHBOARD DE MANTENIMIENTO PREDICTIVO: MACHINE LEARNING + GROQ API
# Challenge Lab - ISOF CLASS
# La app depende unicamente de modelo_mantenimiento.pkl (no lee el CSV).
# ==============================================================================

import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mantenimiento Predictivo", page_icon="🏭", layout="wide")

CARPETA = Path(__file__).resolve().parent
ARCHIVO_MODELO = CARPETA / "modelo_mantenimiento.pkl"

ESTADOS = {
    0: "OPERACIÓN NORMAL",
    1: "MANTENIMIENTO PREVENTIVO",
    2: "FALLO CRÍTICO",
}


# --- 1. Cargar el modelo entrenado (solo el .pkl) ---
@st.cache_resource
def cargar_modelo():
    return joblib.load(ARCHIVO_MODELO)


modelo = cargar_modelo()


# --- 2. Clave de Groq: st.secrets con try/except (y variable de entorno como respaldo) ---
def obtener_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


# --- 3. Orden de trabajo / dictamen con Groq ---
def generar_documento_groq(api_key, estado, lecturas, probabilidad, folio):
    """Devuelve el texto en Markdown generado por Groq. Puede lanzar excepciones."""
    from groq import Groq

    client = Groq(api_key=api_key)

    # Detectar dinamicamente un modelo de chat activo (evita errores 404)
    modelos_activos = [
        m.id
        for m in client.models.list().data
        if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "playai", "embed"])
    ]
    if not modelos_activos:
        raise RuntimeError("No hay modelos de texto disponibles en la cuenta de Groq.")
    modelo_groq = next((m for m in modelos_activos if "llama" in m.lower()), modelos_activos[0])

    sistema = (
        "Eres un Jefe de Mantenimiento Industrial con 20 años de experiencia en plantas "
        "de manufactura automatizada. Respondes en español, con precisión técnica y en Markdown."
    )

    datos = (
        f"- Temperatura: {lecturas['temperatura_c']} °C\n"
        f"- Vibración: {lecturas['vibracion_mms']} mm/s\n"
        f"- Horas de operación: {lecturas['horas_operacion']} hrs\n"
        f"- Presión: {lecturas['presion_psi']} PSI\n"
        f"- Diagnóstico del modelo de ML: {ESTADOS[estado]} (probabilidad {probabilidad:.1f}%)"
    )

    if estado == 0:
        usuario = (
            f"Un modelo predictivo analizó una máquina de la planta:\n{datos}\n\n"
            "Redacta un dictamen BREVE (máximo 5 líneas) de monitoreo preventivo: "
            "confirma que la operación es estable y sugiere una recomendación de rutina."
        )
    else:
        urgencia = "URGENTE, con protocolo de paro inmediato" if estado == 2 else "PREVENTIVA, programada"
        usuario = (
            f"Un modelo predictivo analizó una máquina de la planta:\n{datos}\n\n"
            f"Genera una ORDEN DE TRABAJO {urgencia} en formato Markdown con estas secciones exactas:\n"
            f"1. **Folio:** usa exactamente `{folio}`\n"
            "2. **Diagnóstico:** explica qué lecturas son anómalas y el riesgo asociado.\n"
            "3. **Acciones Técnicas:** exactamente 3 acciones numeradas, concretas y aplicables.\n"
            "4. **Equipo de Seguridad (EPP):** lista del equipo de protección personal requerido."
        )

    respuesta = client.chat.completions.create(
        model=modelo_groq,
        messages=[
            {"role": "system", "content": sistema},
            {"role": "user", "content": usuario},
        ],
        temperature=0.5,
        max_tokens=1024,
    )
    return modelo_groq, respuesta.choices[0].message.content


# --- 4. Interfaz ---
st.title("🏭 Sistema Predictivo de Mantenimiento Industrial")
st.write(
    "Sistema híbrido: **Machine Learning (Random Forest)** clasifica el estado de la máquina "
    "y **Groq API** redacta la orden de trabajo técnica."
)

st.sidebar.header("Lecturas de sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", 40.0, 120.0, 60.0, step=0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 12.0, 1.5, step=0.1)
horas = st.sidebar.slider("Horas de operación", 0, 5000, 300, step=10)
presion = st.sidebar.slider("Presión (PSI)", 70.0, 170.0, 92.0, step=0.5)

# Tarjetas con las lecturas actuales
st.subheader("📟 Lecturas actuales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperatura", f"{temperatura} °C")
col2.metric("Vibración", f"{vibracion} mm/s")
col3.metric("Operación", f"{horas} hrs")
col4.metric("Presión", f"{presion} PSI")

if st.button("🔍 Evaluar máquina y generar documento", type="primary"):
    lecturas = {
        "temperatura_c": temperatura,
        "vibracion_mms": vibracion,
        "horas_operacion": horas,
        "presion_psi": presion,
    }
    entrada = pd.DataFrame([lecturas])

    estado = int(modelo.predict(entrada)[0])
    probas = modelo.predict_proba(entrada)[0]
    prob_por_clase = {int(c): float(p) * 100 for c, p in zip(modelo.classes_, probas)}
    probabilidad = prob_por_clase[estado]

    # --- Diagnóstico del modelo (siempre se muestra, aunque falle Groq) ---
    st.subheader("📊 Diagnóstico del modelo predictivo")
    if estado == 0:
        st.success(f"✅ **{ESTADOS[0]}** — Probabilidad: {probabilidad:.1f}%")
    elif estado == 1:
        st.warning(f"⚠️ **{ESTADOS[1]}** — Probabilidad: {probabilidad:.1f}%")
    else:
        st.error(f"🚨 **{ESTADOS[2]}** — Probabilidad: {probabilidad:.1f}%")

    c0, c1, c2 = st.columns(3)
    c0.metric("P(Normal)", f"{prob_por_clase.get(0, 0.0):.1f}%")
    c1.metric("P(Preventivo)", f"{prob_por_clase.get(1, 0.0):.1f}%")
    c2.metric("P(Crítico)", f"{prob_por_clase.get(2, 0.0):.1f}%")

    # --- Documento con Groq (si falla, la app sigue funcionando) ---
    titulo = "📝 Dictamen de monitoreo" if estado == 0 else "📋 Orden de Trabajo"
    st.subheader(f"{titulo} (Groq API)")

    sufijo = {0: "N", 1: "P", 2: "C"}[estado]
    folio = f"OT-{datetime.now():%Y%m%d-%H%M%S}-{sufijo}"

    api_key = obtener_api_key()
    if not api_key:
        st.info("ℹ️ No hay clave de Groq configurada: se muestra solo el diagnóstico de ML.")
    else:
        try:
            with st.spinner("Generando documento con Groq..."):
                modelo_groq, texto = generar_documento_groq(api_key, estado, lecturas, probabilidad, folio)
            st.caption(f"⚡ Generado con el modelo: `{modelo_groq}`")
            st.markdown(texto)
        except Exception as e:
            st.warning(
                "⚠️ No se pudo generar el documento con Groq (clave inválida, límite de "
                "peticiones o falla de red). El diagnóstico de Machine Learning sigue siendo válido."
            )
            st.caption(f"Detalle técnico: {type(e).__name__}")
