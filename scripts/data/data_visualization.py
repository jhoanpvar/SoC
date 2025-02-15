import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import os
import logging
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

# Configuración del logger para el módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Configurar el nivel de logging según sea necesario

# Crear un manejador de archivos para registrar los logs
fh = logging.FileHandler('data_visualization.log')
fh.setLevel(logging.DEBUG)

# Crear un formato para los logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Añadir el manejador al logger
logger.addHandler(fh)

def cargar_datos_h5(path_archivo):
    """
    Carga los datos desde un archivo HDF5 y los organiza en un diccionario.

    :param path_archivo: Ruta del archivo HDF5.
    :return: Diccionario con los datos cargados.
    """
    dic_lista_dfs = {}
    try:
        with h5py.File(path_archivo, 'r') as hdf:
            for group in hdf.keys():
                dic_lista_dfs[group] = {}
                group_group = hdf[group]
                for cycle_index in group_group.keys():
                    data = group_group[cycle_index][:]
                    columns = list(group_group[cycle_index].attrs['columns'])
                    df = pd.DataFrame(data, columns=columns)
                    dic_lista_dfs[group][int(cycle_index)] = df
        logger.info(f"Datos cargados correctamente desde '{path_archivo}'.")
    except Exception as e:
        logger.error(f"Error al cargar datos desde '{path_archivo}': {e}")
    return dic_lista_dfs

def graficar_ciclos_separados(dic_lista_dfs, grupo, ciclos_seleccionados, salto=1, ruta_guardado=None, guardar_svg=False):
    """
    Genera y guarda gráficos de forma independiente para los ciclos seleccionados.

    :param dic_lista_dfs: Diccionario con los datos cargados.
    :param grupo: Grupo seleccionado.
    :param ciclos_seleccionados: Lista de ciclos a graficar.
    :param salto: Entero que especifica el intervalo entre los números de ciclo a graficar.
    :param ruta_guardado: Ruta donde se guardarán las imágenes (opcional).
    :param guardar_svg: Booleano para indicar si también se guardan en formato SVG.
    """
    if grupo not in dic_lista_dfs:
        logger.warning(f"El grupo '{grupo}' no está disponible en los datos.")
        return

    available_cycles = sorted(dic_lista_dfs[grupo].keys())
    ciclos_validos = sorted([c for c in ciclos_seleccionados if c in available_cycles])

    if not ciclos_validos:
        logger.warning(f"No se encontraron los ciclos especificados en el grupo '{grupo}'.")
        return

    # Aplicar el salto basado en el número de ciclo
    if salto < 1:
        logger.warning(f"Salto inválido: {salto}. Usando salto=1 por defecto.")
        salto = 1

    min_cycle = ciclos_validos[0]
    ciclos_filtrados = [c for c in ciclos_validos if (c - min_cycle) % salto == 0]

    if not ciclos_filtrados:
        logger.warning(f"No hay ciclos que graficar después de aplicar el salto de {salto}.")
        return

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(ciclos_filtrados)))

    if ruta_guardado:
        os.makedirs(ruta_guardado, exist_ok=True)
        logger.info(f"Carpeta creada para guardar los gráficos: '{ruta_guardado}'")

    for tipo in ['Voltage_vs_Capacity', 'Current_vs_Capacity', 'dVdQ_vs_Capacity']:
        plt.figure(figsize=(10, 6))
        for ciclo, color in zip(ciclos_filtrados, colors):
            df = dic_lista_dfs[grupo][ciclo]
            required_columns = ['Charge_Capacity(Ah)', 'Current(A)', 'Voltage(V)', 'dV/dQ']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"El ciclo {ciclo} en el grupo {grupo} no tiene las columnas necesarias. Saltando...")
                continue

            if tipo == 'Voltage_vs_Capacity':
                plt.plot(df['Charge_Capacity(Ah)'], df['Voltage(V)'], label=f'Ciclo {ciclo}', color=to_rgba(color, alpha=0.7))
                plt.xlabel('Charge Capacity (Ah)')
                plt.ylabel('Voltage (V)')
                plt.title(f'Voltaje vs Capacidad - Grupo {grupo}')
            elif tipo == 'Current_vs_Capacity':
                plt.plot(df['Charge_Capacity(Ah)'], df['Current(A)'], label=f'Ciclo {ciclo}', color=to_rgba(color, alpha=0.7))
                plt.xlabel('Charge Capacity (Ah)')
                plt.ylabel('Current (A)')
                plt.title(f'Corriente vs Capacidad - Grupo {grupo}')
            elif tipo == 'dVdQ_vs_Capacity':
                plt.plot(df['Charge_Capacity(Ah)'], df['dV/dQ'], label=f'Ciclo {ciclo}', color=to_rgba(color, alpha=0.7))
                plt.xlabel('Charge Capacity (Ah)')
                plt.ylabel('dV/dQ')
                plt.title(f'dV/dQ vs Capacidad - Grupo {grupo}')

        plt.legend()
        plt.tight_layout()

        if ruta_guardado:
            nombre_archivo = f"{tipo}_grupo_{grupo}.png"
            ruta_png = os.path.join(ruta_guardado, nombre_archivo)
            plt.savefig(ruta_png, dpi=300)
            logger.info(f"Gráfico guardado en '{ruta_png}'")

            if guardar_svg:
                ruta_svg = os.path.join(ruta_guardado, f"{tipo}_grupo_{grupo}.svg")
                plt.savefig(ruta_svg, format='svg')
                logger.info(f"Gráfico SVG guardado en '{ruta_svg}'")

        plt.close()
    logger.info("Generación de gráficos completada.")

def graficar_datos_lista_delta(lista_dfs, grupo, cycle_index, start_idx, end_idx, ruta_guardado=None, guardar_svg=False):
    """
    Genera y guarda gráficos para un subconjunto específico de datos dentro de un ciclo.

    :param lista_dfs: Diccionario con los datos cargados.
    :param grupo: Grupo seleccionado.
    :param cycle_index: Índice del ciclo seleccionado.
    :param start_idx: Índice de inicio del subconjunto.
    :param end_idx: Índice de fin del subconjunto.
    :param ruta_guardado: Ruta donde se guardarán las imágenes (opcional).
    :param guardar_svg: Booleano para indicar si también se guardan en formato SVG.
    """
    try:
        df = lista_dfs[grupo][cycle_index]  # Seleccionar el DataFrame basado en el grupo y el ciclo proporcionados

        # Verificar si las columnas necesarias están presentes
        required_columns = ['Charge_Capacity(Ah)', 'Current(A)', 'Voltage(V)', 'dV/dQ']
        if not all(column in df.columns for column in required_columns):
            raise ValueError("Las columnas requeridas no están presentes en el DataFrame.")

        # Configuración de la figura y los ejes
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Gráfico de Corriente vs. Charge_Capacity(Ah)
        axs[0].plot(df['Charge_Capacity(Ah)'], df['Current(A)'], 'tab:red')
        axs[0].set_xlabel('Charge Capacity (Ah)')
        axs[0].set_ylabel('Current (A)')
        axs[0].set_title(f'Current vs. Charge Capacity - Cycle: {cycle_index}')
        if start_idx is not None and end_idx is not None:
            axs[0].axvline(x=df['Charge_Capacity(Ah)'].iloc[start_idx], color='k', linestyle='--', label='Start')
            axs[0].axvline(x=df['Charge_Capacity(Ah)'].iloc[end_idx], color='k', linestyle='--', label='End')
            axs[0].legend()

        # Gráfico de Tensión vs. Charge_Capacity(Ah)
        axs[1].plot(df['Charge_Capacity(Ah)'], df['Voltage(V)'], 'tab:blue')
        axs[1].set_xlabel('Charge Capacity (Ah)')
        axs[1].set_ylabel('Voltage (V)')
        axs[1].set_title(f'Voltage vs. Charge Capacity - Cycle: {cycle_index}')
        if start_idx is not None and end_idx is not None:
            axs[1].axvline(x=df['Charge_Capacity(Ah)'].iloc[start_idx], color='k', linestyle='--')
            axs[1].axvline(x=df['Charge_Capacity(Ah)'].iloc[end_idx], color='k', linestyle='--')

        # Gráfico de dV/dQ vs. Charge_Capacity(Ah)
        axs[2].plot(df['Charge_Capacity(Ah)'], df['dV/dQ'], 'tab:green')
        axs[2].set_xlabel('Charge Capacity (Ah)')
        axs[2].set_ylabel('dV/dQ')
        axs[2].set_title(f'dV/dQ vs. Charge Capacity - Cycle: {cycle_index}')
        if start_idx is not None and end_idx is not None:
            axs[2].axvline(x=df['Charge_Capacity(Ah)'].iloc[start_idx], color='k', linestyle='--')
            axs[2].axvline(x=df['Charge_Capacity(Ah)'].iloc[end_idx], color='k', linestyle='--')

        plt.tight_layout()

        if ruta_guardado:
            os.makedirs(ruta_guardado, exist_ok=True)
            logger.info(f"Carpeta creada para guardar el gráfico del subconjunto: '{ruta_guardado}'")
            # Mejorar el nombre del archivo para reflejar todos los subplots
            nombre_archivo = f"Subset_Grupo_{grupo}_Ciclo_{cycle_index}.png"
            ruta_png = os.path.join(ruta_guardado, nombre_archivo)
            plt.savefig(ruta_png, dpi=300)
            logger.info(f"Gráfico del subconjunto guardado en '{ruta_png}'")

            if guardar_svg:
                ruta_svg = os.path.join(ruta_guardado, f"Subset_Grupo_{grupo}_Ciclo_{cycle_index}.svg")
                plt.savefig(ruta_svg, format='svg')
                logger.info(f"Gráfico del subconjunto SVG guardado en '{ruta_svg}'")

        plt.show()

    except ValueError as ve:
        logger.error(ve)
        print(ve)
    except KeyError as ke:
        logger.error(f"El ciclo {cycle_index} no existe en el grupo '{grupo}': {ke}")
        print(f"El ciclo {cycle_index} no existe en el grupo '{grupo}'.")
    except Exception as e:
        logger.error(f"Se produjo un error: {e}")
        print(f"Se produjo un error: {e}") 

def mostrar_graficos(grupo, ciclos_seleccionados, salto, ruta_guardado):
    """
    Muestra los gráficos generados en una nueva ventana de Tkinter.

    :param grupo: Grupo seleccionado.
    :param ciclos_seleccionados: Lista de ciclos que se han graficado.
    :param salto: Salto utilizado para la selección de ciclos.
    :param ruta_guardado: Ruta donde se guardaron las imágenes.
    """
    display_window = tk.Toplevel()
    display_window.title("Visualización de Gráficos")
    display_window.geometry("800x600")

    tab_control = ttk.Notebook(display_window)
    tipos = ['Voltage_vs_Capacity', 'Current_vs_Capacity', 'dVdQ_vs_Capacity']
    for tipo in tipos:
        tab = ttk.Frame(tab_control)
        tab_control.add(tab, text=tipo.replace('_', ' ').replace('vs', ' vs '))

        image_path = os.path.join(ruta_guardado or "", f"{tipo}_grupo_{grupo}.png")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                image.thumbnail((780, 580), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                label = ttk.Label(tab, image=photo)
                label.image = photo
                label.pack(expand=True, fill='both')
                logger.info(f"Mostrando imagen: {image_path}")
            except Exception as e:
                label = ttk.Label(tab, text=f"Error al cargar la imagen: {e}")
                label.pack(expand=True, fill='both')
                logger.error(f"Error al cargar la imagen {image_path}: {e}")
        else:
            label = ttk.Label(tab, text="Imagen no encontrada.")
            label.pack(expand=True, fill='both')
            logger.warning(f"Imagen no encontrada: {image_path}")

    tab_control.pack(expand=1, fill='both')
