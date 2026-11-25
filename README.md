# Sistema Predictivo de Mantenimiento Industrial (Challenge Lab ISOF)

Sistema híbrido de **Machine Learning + IA generativa**: un `RandomForestClassifier` clasifica el
estado de una máquina a partir de 4 sensores y la **API de Groq** redacta la orden de trabajo técnica.

| Estado | Significado |
|---|---|
| 0 | Operación normal |
| 1 | Mantenimiento preventivo |
| 2 | Fallo crítico |

## Archivos
- `sensores_planta.csv`: dataset de 30 registros (temperatura, vibración, horas de operación, presión).
- `entrenar.py`: entrena el modelo, imprime el accuracy y exporta `modelo_mantenimiento.pkl`.
- `app.py`: dashboard en Streamlit (usa solo el `.pkl`, no lee el CSV).
- `requirements.txt`: dependencias.

## Uso local
```bash
python -m pip install -r requirements.txt
python entrenar.py
python -m streamlit run app.py
```

## Clave de Groq
La clave **no está en el código**. Se lee con `st.secrets["GROQ_API_KEY"]`.
En local, crea `.streamlit/secrets.toml` (ignorado por git):

```toml
GROQ_API_KEY = "tu_clave"
```

En Streamlit Community Cloud, agrégala en *Advanced settings → Secrets*.
Si la clave falta o es inválida, la app sigue mostrando el diagnóstico de Machine Learning.
