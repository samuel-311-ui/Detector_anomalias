import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Detector de Anomalías — Módulo 4", layout="wide")

st.title("🚨 Detector de Anomalías: Lógica + Big-O + NumPy")
st.caption(
    "Módulo 4 — Matemáticas Discretas y Complejidad. "
    "La misma decisión lógica, evaluada de forma ingenua vs. vectorizada."
)

tab1, tab2, tab3 = st.tabs(
    ["🔎 Simulación de alarma", "📈 Notación Big-O", "⚡ Benchmark en vivo"]
)


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------
# @title
# --- Reto resuelto ---

# Generamos también un dato de "es fin de semana" (booleano) por cada lectura,
# simulando por ejemplo el día de la semana en que se tomó la lectura.
def generar_datos_con_dia(n):
    temperaturas, humedades, finde = generar_datos(n)
    # 0=lunes ... 6=domingo -> fin de semana si es 5 o 6
    dias = np.random.randint(0, 7, n)
    es_fin_de_semana = (dias == 5) | (dias == 6)
    return temperaturas, humedades, es_fin_de_semana


# 1. Versión con loop: P AND Q AND (NOT R)
def alarma_con_loop_v2(temperaturas, humedades, es_fin_de_semana):
    resultados = []
    operaciones = 0
    for temp, hum, fin_semana in zip(temperaturas, humedades, es_fin_de_semana):
        operaciones += 1  # sigue siendo 1 evaluación lógica por dato
        P = temp > 30
        Q = hum < 40
        R = fin_semana
        resultados.append(P and Q and (not R))
    return resultados, operaciones


# 2. Versión vectorizada: misma lógica, con arrays completos
def alarma_vectorizada_v2(temperaturas, humedades, es_fin_de_semana):
    P = temperaturas > 30
    Q = humedades < 40
    R = es_fin_de_semana
    return P & Q & (~R)   # AND y NOT elemento a elemento


# --- Verificación de que ambas versiones coinciden ---
n = 10_000
temps, hums, finde = generar_datos_con_dia(n)

resultados_loop, ops = alarma_con_loop_v2(temps, hums, finde)
resultados_vec = alarma_vectorizada_v2(temps, hums, finde)

coinciden = np.array_equal(np.array(resultados_loop), resultados_vec)
print(f"¿Coinciden loop y vectorizado?: {coinciden}")


# --- Medición con n = 1,000,000 ---
n = 1_000_000
temps_n, hums_n, finde_n = generar_datos_con_dia(n)

inicio = time.time()
resultados_loop, operaciones = alarma_con_loop_v2(temps_n, hums_n, finde_n)
t_loop = time.time() - inicio

inicio = time.time()
resultados_vec = alarma_vectorizada_v2(temps_n, hums_n, finde_n)
t_vec = time.time() - inicio

aceleracion = t_loop / t_vec if t_vec > 0 else float("inf")
print(f"n={n:,}")
print(f"Operaciones lógicas: {operaciones:,}")
print(f"Tiempo loop:  {t_loop:.5f}s")
print(f"Tiempo numpy: {t_vec:.5f}s")
print(f"NumPy es {aceleracion:.1f}x más rápido")

# ---------------------------------------------------------------------------
# Tab 1: Simulación de alarma
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Alarma por regla lógica")
    st.write(
        "La alarma dispara con una condición fija que tú defines, combinando dos "
        "proposiciones con un **AND** lógico: "
        "*temperatura > umbral* **Y** *humedad < umbral*."
    )

    col_cfg, col_data = st.columns([1, 2])

    with col_cfg:
        n = st.slider("Número de lecturas (n)", 50, 5000, 500, step=50)
        temp_umbral = st.slider("Umbral temperatura (°C) — mayor que", 15, 40, 30)
        hum_umbral = st.slider("Umbral humedad (%) — menor que", 20, 80, 40)

    temps, hums = generar_datos(n)

    with col_cfg:
        alarmas = alarma_logica_vectorizada(temps, hums, temp_umbral, hum_umbral)
        st.metric("Alarmas detectadas", f"{alarmas.sum()} / {n}")

    with col_data:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(
            temps[~alarmas], hums[~alarmas],
            c="steelblue", alpha=0.5, label="Normal", s=15,
        )
        ax.scatter(
            temps[alarmas], hums[alarmas],
            c="crimson", alpha=0.8, label="Alarma / anomalía", s=25,
        )
        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel("Humedad (%)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with st.expander("Ver datos y lógica aplicada"):
        df = pd.DataFrame({
            "temperatura": temps.round(2),
            "humedad": hums.round(2),
            "alarma": alarmas,
        })
        st.dataframe(df, use_container_width=True, height=250)


# ---------------------------------------------------------------------------
# Tab 2: Notación Big-O
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("¿Por qué importa la complejidad?")
    st.write(
        "El detector de arriba recorre `n` lecturas una vez: es un algoritmo **O(n)**. "
        "Aquí puedes ver qué tan distinto crece el número de operaciones frente a otras "
        "complejidades comunes, a medida que aumentan los datos."
    )

    n_max = st.slider("Tamaño máximo de n para la gráfica", 10, 200, 50)
    n_valores = np.arange(1, n_max + 1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(n_valores, np.ones_like(n_valores), label="O(1) — constante")
    ax2.plot(n_valores, n_valores, label="O(n) — lineal (nuestro detector)")
    ax2.plot(n_valores, n_valores * np.log2(np.maximum(n_valores, 2)), label="O(n log n)")
    ax2.plot(n_valores, n_valores ** 2, label="O(n²) — cuadrática")
    ax2.set_xlabel("Tamaño de los datos (n)")
    ax2.set_ylabel("Operaciones (teórico)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    st.info(
        "Nuestro detector de alarmas es O(n) tanto con loop como con NumPy: "
        "la notación no cambia. Lo que cambia es la **constante** detrás de cada operación, "
        "y eso es justo lo que exploras en la siguiente pestaña."
    )


# ---------------------------------------------------------------------------
# Tab 3: Benchmark en vivo
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Loop vs. NumPy: misma lógica, distinta velocidad real")
    st.write(
        "Ejecuta la misma condición lógica sobre datos sintéticos, una vez con un loop "
        "de Python puro y otra vez vectorizada con NumPy, y compara el tiempo real."
    )

    n_bench = st.select_slider(
        "Tamaño de datos para el benchmark",
        options=[1_000, 10_000, 100_000, 500_000, 1_000_000],
        value=100_000,
    )
    temp_umbral_b = st.slider("Umbral temperatura (°C)", 15, 40, 30, key="temp_bench")
    hum_umbral_b = st.slider("Umbral humedad (%)", 20, 80, 40, key="hum_bench")

    if st.button("▶️ Ejecutar benchmark", type="primary"):
        temps_b, hums_b = generar_datos(n_bench)

        # perf_counter tiene mucha más resolución que time.time(), y repetimos
        # varias veces porque la versión vectorizada puede ser demasiado rápida
        # para medirse de forma confiable en una sola corrida.
        repeticiones_loop = 1
        repeticiones_vec = 20

        inicio = time.perf_counter()
        for _ in range(repeticiones_loop):
            alarma_logica_loop(temps_b, hums_b, temp_umbral_b, hum_umbral_b)
        t_loop = (time.perf_counter() - inicio) / repeticiones_loop

        inicio = time.perf_counter()
        for _ in range(repeticiones_vec):
            alarma_logica_vectorizada(temps_b, hums_b, temp_umbral_b, hum_umbral_b)
        t_vec = (time.perf_counter() - inicio) / repeticiones_vec

        col1, col2, col3 = st.columns(3)
        col1.metric("Tiempo con loop", f"{t_loop*1000:.3f} ms")
        col2.metric("Tiempo con NumPy", f"{t_vec*1000:.3f} ms")

        if t_vec > 0:
            speedup = t_loop / t_vec
            col3.metric("NumPy es más rápido por", f"{speedup:,.0f}x")
        else:
            col3.metric("NumPy es más rápido por", "demasiado rápido para medir")

        st.caption(
            f"Tiempo con loop promediado sobre {repeticiones_loop} corrida(s); "
            f"tiempo con NumPy promediado sobre {repeticiones_vec} corridas, "
            "para reducir el ruido de la medición."
        )

        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.bar(["Loop (Python)", "NumPy (vectorizado)"],
                [t_loop * 1000, t_vec * 1000],
                color=["indianred", "seagreen"])
        ax3.set_ylabel("Tiempo (milisegundos)")
        ax3.grid(alpha=0.3, axis="y")
        st.pyplot(fig3)
    else:
        st.caption("Ajusta los parámetros y presiona **Ejecutar benchmark** para ver el resultado.")
