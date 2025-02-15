import pandas as pd
import matplotlib.pyplot as plt

# Ruta del archivo CSV
file_path = "ruta_al_archivo/metricas.csv"  # Reemplaza con el camino real

# Cargar los datos del archivo CSV
data_csv = pd.read_csv(file_path)

# Multiplicar los valores de RMSE por 100
data_csv["RMSE"] = data_csv["RMSE"] * 100

# Obtener segmentos y modelos únicos
segments = data_csv["Segmento"].unique()
models = data_csv["Modelo"].unique()
colors = ['steelblue', 'orange', 'gray']  # Colores para los modelos

# Generar gráficos para cada segmento
for segment in segments:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filtrar los datos para el segmento actual
    segment_data = data_csv[data_csv["Segmento"] == segment]

    # Pivotar los datos para organizar las métricas por número de datos y modelo
    pivot_data = segment_data.pivot(index="Número de Datos", columns="Modelo", values="RMSE")
    pivot_data = pivot_data[models]  # Asegurar el orden de los modelos

    # Crear el gráfico de barras
    pivot_data.plot(kind='bar', ax=ax, color=colors)

    # Configuración del gráfico
    ax.set_xlabel("Número de dados")
    ax.set_ylabel("RMSE (%)")
    ax.legend(title="Modelos")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Guardar el gráfico como SVG con la nomenclatura especificada
    file_name = f"RMSE_ALL_Seg{segment}.svg"
    plt.tight_layout()
    plt.savefig(file_name, format='svg')

    # Mostrar el gráfico
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Ruta de los archivos CSV
file_paths = {
    "Model 2 Segment A Data55": "ruta_a/Model_2_Segment_A_Data55_losses.csv",
    "Model 3 Segment B Data60": "ruta_a/Model_3_Segment_B_Data60_losses.csv",
    "Model 3 Segment C Data55": "ruta_a/Model_3_Segment_C_Data55_losses.csv",
    "Model 3 Segment F Data55": "ruta_a/Model_3_Segment_F_Data55_losses.csv"
}

# Colores para las curvas
colors = {"Training Loss": "blue", "Validation Loss": "orange"}

# Generar gráficos
for model_name, path in file_paths.items():
    # Leer los datos del archivo
    data = pd.read_csv(path)

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["Epoch"], data["Train Loss"], label="Perda de Treinamento", color=colors["Training Loss"])
    ax.plot(data["Epoch"], data["Val Loss"], label="Perda de Validação", color=colors["Validation Loss"])

    # Configurar el gráfico
    ax.set_title(f"Curvas de Perda - {model_name}")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Perda")
    ax.legend()
    ax.grid(alpha=0.5)

    # Guardar el gráfico como SVG
    file_name = f"{model_name.replace(' ', '_')}_losses.svg"
    plt.tight_layout()
    plt.savefig(file_name, format="svg")
    plt.show()
