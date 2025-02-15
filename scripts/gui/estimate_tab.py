# estimate_tab.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import torch
import h5py
from scripts.models.model_training import (
    create_model
)
from scripts.data.data_preparation import (
    cargar_diccionario_desde_hdf5
)
import random
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from threading import Lock

class EstimateTab:
    def __init__(self, parent, custom_model_params):
        self.parent = parent
        self.custom_model_params = custom_model_params
        self.loaded_models = {}  # Diccionario para almacenar modelos por segmento
        self.dic_lista_dfs = {}  # Diccionario para almacenar los datos cargados
        self.estadisticas_globales = {}  # Diccionario para estadísticas globales
        self.logger = logging.getLogger('EstimateTab')
        self.device = torch.device('cpu')  # Usar CPU por defecto
        self.loaded_models_lock = Lock()  # Bloqueo para loaded_models
        self.create_widgets()

    def create_widgets(self):
        # Crear Notebook para las diferentes pestañas
        self.notebook = ttk.Notebook(self.parent, padding=10)
        self.notebook.pack(fill='both', expand=True)

        # Pestaña de Estimación
        self.estimate_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.estimate_frame, text='Estimación')

        # Pestaña de Visualización de Pérdidas
        self.loss_plot_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.loss_plot_frame, text='Visualizar Pérdidas')

        # Pestaña de Métricas
        self.metrics_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.metrics_frame, text='Métricas')

        # Widgets para la pestaña de Estimación
        self.create_estimate_widgets()

        # Widgets para la pestaña de Visualización de Pérdidas
        self.create_loss_plot_widgets()

        # Widgets para la pestaña de Métricas
        self.create_metrics_widgets()

    def create_estimate_widgets(self):
        # --- Selección de Directorio de Modelos ---
        self.models_dir_label = ttk.Label(self.estimate_frame, text="Directorio de Modelos:")
        self.models_dir_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.models_dir_entry = ttk.Entry(self.estimate_frame, width=50)
        self.models_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky='we')

        self.models_dir_button = ttk.Button(
            self.estimate_frame, text="Seleccionar Directorio",
            command=self.select_models_directory
        )
        self.models_dir_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Selección del Archivo de Datos (.h5) ---
        self.data_file_label = ttk.Label(self.estimate_frame, text="Archivo de Datos (.h5):")
        self.data_file_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

        self.data_file_entry = ttk.Entry(self.estimate_frame, width=50)
        self.data_file_entry.grid(row=1, column=1, padx=5, pady=5, sticky='we')

        self.data_file_button = ttk.Button(
            self.estimate_frame, text="Seleccionar Archivo de Datos",
            command=self.select_data_file
        )
        self.data_file_button.grid(row=1, column=2, padx=5, pady=5)

        # --- Botón para Cargar Modelos y Datos ---
        self.load_all_button = ttk.Button(
            self.estimate_frame, text="Cargar Modelos y Datos",
            command=self.load_all_models_and_data
        )
        self.load_all_button.grid(row=2, column=1, padx=5, pady=10)

        # --- Botón para Realizar Comparación de Estimaciones ---
        self.compare_button = ttk.Button(
            self.estimate_frame, text="Comparar Estimaciones",
            command=self.compare_estimations, state='disabled'
        )
        self.compare_button.grid(row=3, column=1, padx=5, pady=5)

        # --- Treeview para mostrar los resultados ---
        self.results_tree = ttk.Treeview(self.estimate_frame, columns=("Parámetro", "Valor"), show='headings', height=8)
        self.results_tree.heading("Parámetro", text="Parámetro")
        self.results_tree.heading("Valor", text="Valor")
        self.results_tree.column("Parámetro", width=200, anchor='center')
        self.results_tree.column("Valor", width=200, anchor='center')
        self.results_tree.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky='we')

        # Scrollbar para el Treeview
        self.scrollbar = ttk.Scrollbar(self.estimate_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscroll=self.scrollbar.set)
        self.scrollbar.grid(row=4, column=3, sticky='ns', pady=5)

        # Etiqueta de estado
        self.status_label2 = ttk.Label(self.estimate_frame, text="")
        self.status_label2.grid(row=5, column=0, columnspan=4, padx=5, pady=5, sticky='w')

    def clear_results_tree(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def create_loss_plot_widgets(self):
        # Seleccionar archivo CSV con las pérdidas
        self.loss_file_label = ttk.Label(self.loss_plot_frame, text="Seleccionar CSV de Pérdidas:")
        self.loss_file_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.loss_file_entry = ttk.Entry(self.loss_plot_frame, width=50)
        self.loss_file_entry.grid(row=0, column=1, padx=5, pady=5, sticky='we')

        self.loss_file_button = ttk.Button(
            self.loss_plot_frame, text="Seleccionar CSV",
            command=self.select_loss_file
        )
        self.loss_file_button.grid(row=0, column=2, padx=5, pady=5)

        # Checkboxes para seleccionar formatos de guardado
        self.save_png_var = tk.BooleanVar()
        self.save_svg_var = tk.BooleanVar()
        self.save_png_check = ttk.Checkbutton(
            self.loss_plot_frame, text="Guardar como PNG", variable=self.save_png_var
        )
        self.save_png_check.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        self.save_svg_check = ttk.Checkbutton(
            self.loss_plot_frame, text="Guardar como SVG", variable=self.save_svg_var
        )
        self.save_svg_check.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Entrada para el directorio de guardado
        self.save_dir_label = ttk.Label(self.loss_plot_frame, text="Directorio de Guardado:")
        self.save_dir_label.grid(row=2, column=0, padx=5, pady=5, sticky='e')

        self.save_dir_entry = ttk.Entry(self.loss_plot_frame, width=50, state="disabled")
        self.save_dir_entry.grid(row=2, column=1, padx=5, pady=5, sticky='we')

        self.save_dir_button = ttk.Button(
            self.loss_plot_frame, text="Seleccionar Directorio",
            command=self.select_save_directory, state="disabled"
        )
        self.save_dir_button.grid(row=2, column=2, padx=5, pady=5)

        # Botón para cargar y mostrar el gráfico
        self.plot_button = ttk.Button(
            self.loss_plot_frame, text="Mostrar Gráfico",
            command=self.plot_losses, state='disabled'
        )
        self.plot_button.grid(row=3, column=1, padx=5, pady=10)

        # Opciones para visualizar los gráficos
        self.display_var = tk.BooleanVar()
        self.display_checkbutton = ttk.Checkbutton(
            self.loss_plot_frame, text="Mostrar Gráficos en la Aplicación",
            variable=self.display_var
        )
        self.display_checkbutton.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        # Etiqueta de estado
        self.status_label3 = ttk.Label(self.loss_plot_frame, text="")
        self.status_label3.grid(row=5, column=0, columnspan=4, padx=5, pady=5, sticky='w')

        # Habilitar o deshabilitar botones según los checkboxes
        self.save_png_var.trace_add('write', self.toggle_save_options)
        self.save_svg_var.trace_add('write', self.toggle_save_options)

    def create_metrics_widgets(self):
        # --- Selección del Número de Iteraciones ---
        self.iterations_label = ttk.Label(self.metrics_frame, text="Número de Iteraciones:")
        self.iterations_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.iterations_entry = ttk.Entry(self.metrics_frame, width=20)
        self.iterations_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.iterations_entry.insert(0, "100")  # Valor por defecto

        # --- Botón para Iniciar Cálculo de Métricas ---
        self.calculate_metrics_button = ttk.Button(
            self.metrics_frame, text="Calcular Métricas",
            command=self.calculate_metrics, state='disabled'
        )
        self.calculate_metrics_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Treeview para Mostrar Resultados de RMSE ---
        self.metrics_tree = ttk.Treeview(self.metrics_frame, columns=("Métrica", "Valor"), show='headings', height=4)
        self.metrics_tree.heading("Métrica", text="Métrica")
        self.metrics_tree.heading("Valor", text="Valor")
        self.metrics_tree.column("Métrica", width=200, anchor='center')
        self.metrics_tree.column("Valor", width=200, anchor='center')
        self.metrics_tree.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky='we')

        # Scrollbar para el Treeview de métricas
        self.metrics_scrollbar = ttk.Scrollbar(self.metrics_frame, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscroll=self.metrics_scrollbar.set)
        self.metrics_scrollbar.grid(row=1, column=3, sticky='ns', pady=5)

        # Etiqueta de estado para métricas
        self.metrics_status_label = ttk.Label(self.metrics_frame, text="")
        self.metrics_status_label.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky='w')

    def select_models_directory(self):
        models_dir = filedialog.askdirectory(
            title="Seleccionar Directorio de Modelos"
        )
        if models_dir:
            self.models_dir_entry.delete(0, tk.END)
            self.models_dir_entry.insert(0, models_dir)

    def select_data_file(self):
        data_file = filedialog.askopenfilename(
            filetypes=[("HDF5 Files", "*.h5")],
            title="Seleccionar Archivo de Datos de Prueba"
        )
        if data_file:
            self.data_file_entry.delete(0, tk.END)
            self.data_file_entry.insert(0, data_file)

    def load_all_models_and_data(self):
        models_dir = self.models_dir_entry.get()
        if not os.path.isdir(models_dir):
            messagebox.showerror("Error", "Seleccione un directorio de modelos válido.")
            return

        data_file = self.data_file_entry.get()
        if not os.path.isfile(data_file):
            messagebox.showerror("Error", "Seleccione un archivo de datos válido (.h5).")
            return

        self.status_label2.config(text="Cargando modelos y datos...")
        self.compare_button.config(state='disabled')
        self.calculate_metrics_button.config(state='disabled')
        threading.Thread(target=self.load_all_models_and_data_thread, args=(models_dir, data_file), daemon=True).start()

    def load_all_models_and_data_thread(self, models_dir, data_file):
        try:
            with self.loaded_models_lock:
                # Cargar modelos desde el directorio
                for filename in os.listdir(models_dir):
                    if filename.endswith(".pth"):
                        filepath = os.path.join(models_dir, filename)
                        parts = filename.split('_Segment_')
                        if len(parts) != 2:
                            self.logger.warning(f"Nombre de archivo no reconocido: {filename}. Saltando.")
                            continue
                        model_name_part, rest = parts
                        segment_part, data_part = rest.split('_Data')
                        segment = segment_part.strip().upper()  # Normalizar a mayúsculas
                        try:
                            numero_de_datos = int(data_part.strip().replace('.pth', ''))
                        except ValueError:
                            self.logger.warning(f"No se pudo extraer numero_de_datos de {filename}. Saltando.")
                            continue

                        self.logger.info(f"Cargando modelo para segmento {segment}: {model_name_part}")

                        # Cargar parámetros personalizados si corresponde
                        if model_name_part.startswith('Modelo Personalizado'):
                            params = self.custom_model_params.get(model_name_part)
                            if not params:
                                messagebox.showerror("Error", f"No se encontraron parámetros para {model_name_part}.")
                                self.status_label2.config(text="")
                                return
                        else:
                            params = None

                        # Crear y cargar el modelo
                        model = create_model(model_name_part, input_channels=3, custom_params=params, device=self.device)
                        model.load_state_dict(torch.load(filepath, map_location=self.device))
                        model.to(self.device)  # Mover al dispositivo
                        model.eval()
                        self.loaded_models[segment] = {
                            'model': model,
                            'numero_de_datos': numero_de_datos
                        }
                        self.logger.info(f"Modelo para segmento {segment} cargado exitosamente.")

                # Verificación de modelos requeridos
                required_segments = ['A', 'B', 'C', 'F']
                missing_segments = [seg for seg in required_segments if seg not in self.loaded_models]
                if missing_segments:
                    self.logger.error(f"Faltan modelos para los segmentos: {missing_segments}")
                    messagebox.showerror("Error", f"Faltan modelos para los segmentos: {', '.join(missing_segments)}")
                    self.status_label2.config(text="")
                    return

                self.logger.info(f"Modelos cargados: {list(self.loaded_models.keys())}")

            # Cargar datos y estadísticas globales
            self.dic_lista_dfs, self.estadisticas_globales = cargar_diccionario_desde_hdf5(data_file)
            self.logger.info(f"Datos cargados desde {data_file}")

            # Verificar disponibilidad de segmentos
            segmentos_cargados = list(self.loaded_models.keys())
            self.logger.info(f"Segmentos cargados: {segmentos_cargados}")

            self.status_label2.config(text="Modelos y datos cargados con éxito.")
            self.compare_button.config(state='normal')
            self.calculate_metrics_button.config(state='normal')
        except Exception as e:
            self.status_label2.config(text="")
            messagebox.showerror("Error", f"Se produjo un error al cargar los modelos o datos: {e}")
            self.logger.error(f"Error al cargar modelos o datos: {e}")

    def verify_model_segment(self, segment, model):
        """
        Verifica que el modelo proporcionado corresponde al segmento especificado.

        Args:
            segment (str): El segmento al que se supone que debe corresponder el modelo.
            model (torch.nn.Module): El modelo que se va a verificar.

        Returns:
            bool: True si el modelo corresponde al segmento, False en caso contrario.
        """
        with self.loaded_models_lock:
            model_info = self.loaded_models.get(segment)

        if not model_info:
            self.logger.error(f"No se encontró un modelo cargado para el segmento '{segment}'.")
            return False

        # Comparar referencias de objetos de modelos
        if model_info['model'] is not model:
            self.logger.error(f"El modelo para el segmento '{segment}' no coincide con el modelo seleccionado.")
            return False

        self.logger.info(f"Verificación exitosa: El modelo para el segmento '{segment}' es correcto.")
        return True

    def find_closest_range(self, charge_capacity_test, ranges):
        """
        Encuentra el rango más cercano para un valor dado de 'Charge_Capacity(Ah)'.

        Args:
            charge_capacity_test (float): El valor de 'Charge_Capacity(Ah)' a asignar.
            ranges (dict): Diccionario de rangos definidos.

        Returns:
            str: El nombre del rango más cercano.
        """
        closest_rango = None
        min_distance = float('inf')
        for rango, (low, high) in ranges.items():
            # Calcular la distancia al rango
            if charge_capacity_test < low:
                distance = low - charge_capacity_test
            elif charge_capacity_test > high:
                distance = charge_capacity_test - high
            else:
                distance = 0  # Dentro del rango

            if distance < min_distance:
                min_distance = distance
                closest_rango = rango

        return closest_rango


    def compare_estimations(self):
        self.compare_button.config(state='disabled')
        threading.Thread(target=self.compare_estimations_thread, daemon=True).start()

    def compare_estimations_thread(self):
        try:
            # Definir los grupos objetivo
            grupos_objetivo = ['35', '36', '37', '38']

            # Verificar qué grupos están disponibles
            grupos_disponibles = [g for g in grupos_objetivo if g in self.dic_lista_dfs]
            if not grupos_disponibles:
                raise ValueError(f"Ninguno de los grupos {grupos_objetivo} está disponible en los datos.")

            # Seleccionar un grupo aleatorio de los disponibles
            group = random.choice(grupos_disponibles)
            self.logger.info(f"Grupo seleccionado aleatoriamente: {group}")

            # Obtener datos del primer ciclo
            if 50 in self.dic_lista_dfs[group]:
                df_first_cycle = self.dic_lista_dfs[group][50]
            else:
                # Si el ciclo 0 no está disponible, usar el ciclo mínimo disponible
                available_cycles = list(self.dic_lista_dfs[group].keys())
                min_cycle = min(available_cycles)
                df_first_cycle = self.dic_lista_dfs[group][min_cycle]
                self.logger.warning(f"El ciclo 0 no está disponible en el grupo {group}. Usando ciclo {min_cycle}.")

            # Definir rangos basados en los cuantiles de 'Charge_Capacity(Ah)' del primer ciclo
            charge_capacities = df_first_cycle['Charge_Capacity(Ah)']
            low_quantile = charge_capacities.quantile(1/3)
            high_quantile = charge_capacities.quantile(2/3)
            ranges = {
                'A': (charge_capacities.min(), low_quantile),
                'B': (low_quantile, high_quantile),
                'C': (high_quantile, charge_capacities.max())
            }
            self.logger.info(f"Rangos establecidos basados en 'Charge_Capacity(Ah)': {ranges}")

            # Obtener rangos para 'Charge_Capacity(Ah)' para el gráfico
            capacities = charge_capacities  # ya definido
            capacity_low_quantile = low_quantile
            capacity_high_quantile = high_quantile
            capacities_ranges = {
                'A': (capacities.min(), capacity_low_quantile),
                'B': (capacity_low_quantile, capacity_high_quantile),
                'C': (capacity_high_quantile, capacities.max())
            }

            # Seleccionar un ciclo aleatorio entre 0 y 100
            cycle = random.randint(0, 100)
            self.logger.info(f"Ciclo seleccionado aleatoriamente: {cycle}")

            # Verificar si el ciclo existe en el grupo seleccionado
            if cycle not in self.dic_lista_dfs[group]:
                raise ValueError(f"El ciclo {cycle} no existe en el grupo {group}.")

            df_cycle = self.dic_lista_dfs[group][cycle]

            # Seleccionar un valor aleatorio de Voltage(V) del ciclo actual
            V_test = random.choice(df_cycle['Voltage(V)'].tolist())
            charge_capacity_test = df_cycle.loc[df_cycle['Voltage(V)'] == V_test, 'Charge_Capacity(Ah)'].iloc[0]
            start_range = df_cycle[df_cycle['Charge_Capacity(Ah)'] == charge_capacity_test].index[0]
            self.logger.info(f"V_test seleccionado: {V_test}, Charge_Capacity_test: {charge_capacity_test}, start_range: {start_range}")

            # Determinar en qué rango está charge_capacity_test
            rango_de_Charge = None
            for rango, (low, high) in ranges.items():
                if rango != 'C' and low <= charge_capacity_test < high:
                    rango_de_Charge = rango
                    break
                elif rango == 'C' and high <= charge_capacity_test <= capacities.max():
                    rango_de_Charge = rango
                    break

            if not rango_de_Charge:
                # Asignar al rango más cercano
                rango_de_Charge = self.find_closest_range(charge_capacity_test, ranges)
                self.logger.warning(f"Charge_Capacity_test {charge_capacity_test} no cae en ninguno de los rangos A, B, C. Asignado al rango más cercano: {rango_de_Charge}")

            self.logger.info(f"Charge_Capacity_test {charge_capacity_test} cae en el rango {rango_de_Charge}")

            # Obtener el modelo correspondiente al rango_de_Charge
            with self.loaded_models_lock:
                model_info = self.loaded_models.get(rango_de_Charge)
            if not model_info:
                raise ValueError(f"No se encontró el modelo para el segmento {rango_de_Charge}.")

            model_selected = model_info['model']
            numero_de_datos = model_info['numero_de_datos']
            self.logger.info(f"Usando modelo {rango_de_Charge} con numero_de_datos: {numero_de_datos}")

            # Verificación de que el modelo seleccionado corresponde al segmento
            if not self.verify_model_segment(rango_de_Charge, model_selected):
                raise ValueError(f"El modelo seleccionado para el segmento '{rango_de_Charge}' no es el correcto.")

            # Verificar si start_range es mayor que numero_de_datos
            if start_range <= numero_de_datos:
                self.logger.warning(f"start_range ({start_range}) no es mayor que numero_de_datos ({numero_de_datos}). Seleccionando otro V_test.")
                # Intentar seleccionar otro V_test hasta cumplir la condición
                ciclos_disponibles = df_cycle[df_cycle.index > numero_de_datos]
                if ciclos_disponibles.empty:
                    raise ValueError("No se pudo encontrar un V_test que cumpla con la condición start_range > numero_de_datos.")
                V_test = random.choice(ciclos_disponibles['Voltage(V)'].tolist())
                charge_capacity_test = df_cycle.loc[df_cycle['Voltage(V)'] == V_test, 'Charge_Capacity(Ah)'].iloc[0]
                start_range = df_cycle[df_cycle['Charge_Capacity(Ah)'] == charge_capacity_test].index[0]
                self.logger.info(f"Nuevo V_test seleccionado: {V_test}, Charge_Capacity_test: {charge_capacity_test}, start_range: {start_range}")

                # Determinar nuevamente el rango de 'Charge_Capacity(Ah)'
                rango_de_Charge = None
                for rango, (low, high) in ranges.items():
                    if rango != 'C' and low <= charge_capacity_test < high:
                        rango_de_Charge = rango
                        break
                    elif rango == 'C' and high <= charge_capacity_test <= capacities.max():
                        rango_de_Charge = rango
                        break

                if not rango_de_Charge:
                    # Asignar al rango más cercano
                    rango_de_Charge = self.find_closest_range(charge_capacity_test, ranges)
                    self.logger.warning(f"Charge_Capacity_test {charge_capacity_test} no cae en ninguno de los rangos A, B, C. Asignado al rango más cercano: {rango_de_Charge}")

                self.logger.info(f"Charge_Capacity_test {charge_capacity_test} cae en el rango {rango_de_Charge}")

                # Obtener el modelo correspondiente al rango_de_Charge
                with self.loaded_models_lock:
                    model_info = self.loaded_models.get(rango_de_Charge)
                if not model_info:
                    raise ValueError(f"No se encontró el modelo para el segmento {rango_de_Charge}.")

                model_selected = model_info['model']
                numero_de_datos = model_info['numero_de_datos']
                self.logger.info(f"Usando modelo {rango_de_Charge} con numero_de_datos: {numero_de_datos}")

                # Verificación de que el modelo seleccionado corresponde al segmento
                if not self.verify_model_segment(rango_de_Charge, model_selected):
                    raise ValueError(f"El modelo seleccionado para el segmento '{rango_de_Charge}' no es el correcto.")

            # Definir 'end_value' como 'charge_capacity_test'
            end_value = charge_capacity_test

            end_range = start_range
            start_range = end_range - numero_de_datos
            start_range = max(start_range, 0)  # Evitar índices negativos
            self.logger.info(f"start_range ajustado: {start_range}, end_range: {end_range}")

            # Construir DataFrame para el rango seleccionado
            df_selected = df_cycle.iloc[start_range:end_range].copy()

            V_vector_selected_normalized = df_selected['Voltage(V)'].values
            Current_vector_selected_normalized = df_selected['Current(A)'].values
            dVdQ_vector_selected_normalized = df_selected['dV/dQ'].values

            # Convertir a tensores y mover al dispositivo
            try:
                X_selected = torch.tensor([
                    V_vector_selected_normalized,
                    Current_vector_selected_normalized,
                    dVdQ_vector_selected_normalized
                ], dtype=torch.float32).unsqueeze(0).to(self.device)
                self.logger.info("Tensor X_selected creado exitosamente.")
            except Exception as e:
                raise ValueError(f"Error al convertir los datos a tensores para el modelo {rango_de_Charge}: {e}")

            # Obtener el modelo F
            with self.loaded_models_lock:
                model_f_info = self.loaded_models.get('F')
            if not model_f_info:
                raise ValueError("No se encontró el modelo para el segmento F.")

            model_f = model_f_info['model']
            numero_de_datos_f = model_f_info['numero_de_datos']
            self.logger.info(f"Usando modelo F con numero_de_datos: {numero_de_datos_f}")

            # Construir DataFrame para el modelo F
            df_f = df_cycle.iloc[start_range:end_range].copy()

            # Extraer los vectores normalizados (ya están normalizados)
            V_vector_f_normalized = df_f['Voltage(V)'].values
            Current_vector_f_normalized = df_f['Current(A)'].values
            dVdQ_vector_f_normalized = df_f['dV/dQ'].values

            # Convertir a tensores para el modelo F y mover al dispositivo
            try:
                X_f = torch.tensor([
                    V_vector_f_normalized,
                    Current_vector_f_normalized,
                    dVdQ_vector_f_normalized
                ], dtype=torch.float32).unsqueeze(0).to(self.device)
                self.logger.info("Tensor X_f creado exitosamente.")
            except Exception as e:
                raise ValueError(f"Error al convertir los datos a tensores para el modelo F: {e}")

            # Predicción del modelo seleccionado (normalizada)
            with torch.no_grad():
                prediction_selected_normalized = model_selected(X_selected).item()

            self.logger.info(f"SoC predicha por el modelo {rango_de_Charge}: {prediction_selected_normalized}")

            # Predicción del modelo F (normalizada)
            with torch.no_grad():
                prediction_f_normalized = model_f(X_f).item()

            self.logger.info(f"SoC predicha por el modelo F: {prediction_f_normalized}")

            # --- Generación del Gráfico Voltage_vs_Capacity ---
            try:
                self.logger.info("Generando gráfico Voltage_vs_Capacity.")
                plt.figure(figsize=(10, 6))
                plt.plot(df_cycle['Charge_Capacity(Ah)'], df_cycle['Voltage(V)'], label='Voltage vs Capacity', color='blue')

                # Línea discontinua en el punto seleccionado
                plt.axvline(x=end_value, color='gray', linestyle='--', label='Punto Seleccionado')
                plt.axhline(y=V_test, color='gray', linestyle='--')

                # Líneas punteadas para los rangos de 'Charge_Capacity(Ah)'
                plt.axvline(x=capacities_ranges['A'][1], color='purple', linestyle='--', label='Límite Rango A/B')
                plt.axvline(x=capacities_ranges['B'][1], color='orange', linestyle='--', label='Límite Rango B/C')

                # Anotaciones de los segmentos A, B, C en el eje x
                plt.text((capacities_ranges['A'][0] + capacities_ranges['A'][1]) / 2, min(df_cycle['Voltage(V)']) - 0.05, 'Segmento A', ha='center', va='top', color='purple')
                plt.text((capacities_ranges['B'][0] + capacities_ranges['B'][1]) / 2, min(df_cycle['Voltage(V)']) - 0.05, 'Segmento B', ha='center', va='top', color='orange')
                plt.text((capacities_ranges['C'][0] + capacities_ranges['C'][1]) / 2, min(df_cycle['Voltage(V)']) - 0.05, 'Segmento C', ha='center', va='top', color='green')

                # Anotaciones de SoC estimadas
                plt.scatter(end_value, V_test, color='red', zorder=5)
                plt.text(end_value, V_test + 0.05, f"Predicción {rango_de_Charge}: {prediction_selected_normalized:.2f} %", color='red')
                plt.text(end_value, V_test - 0.1, f"SoC Predicho (F): {prediction_f_normalized:.2f} %", color='green')
                plt.text(end_value, min(df_cycle['Voltage(V)']) - 0.1, f"SoC Real: {end_value:.2f} %", ha='center', va='top', color='black')

                plt.xlabel('Charge_Capacity (Ah)')
                plt.ylabel('Voltage (V)')
                plt.title(f'Voltage vs Capacity - Grupo {group}, Ciclo {cycle}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                # Mostrar el gráfico
                plt.show()

                self.logger.info("Gráfico Voltage_vs_Capacity generado exitosamente.")
            except Exception as e:
                self.logger.error(f"Error al generar el gráfico Voltage_vs_Capacity: {e}")
                raise ValueError(f"Error al generar el gráfico Voltage_vs_Capacity: {e}")

            # --- Mostrar el resultado en la interfaz usando Treeview ---
            self.clear_results_tree()  # Limpiar resultados anteriores

            # Definir los parámetros y sus valores
            results = {
                "Grupo": group,
                "Ciclo": cycle,
                "V_test (V)": f"{V_test*4.2:.4f}",
                "Charge_Capacity_test (Ah)": f"{charge_capacity_test*1.1:.2f}",
                "Rango": rango_de_Charge,
                "SoC Real (%)": f"{end_value:.2f}",
                f"SoC Predicho ({rango_de_Charge}) (%)": f"{prediction_selected_normalized:.2f}",
                "SoC Predicho (F) (%)": f"{prediction_f_normalized:.2f}"
            }

            # Insertar cada parámetro en el Treeview
            for parametro, valor in results.items():
                self.results_tree.insert("", "end", values=(parametro, valor))

            self.logger.info("Comparación de estimaciones completada.")
        except Exception as e:
            messagebox.showerror("Error", f"Se produjo un error al comparar las estimaciones: {e}")
            self.logger.error(f"Error al comparar las estimaciones: {e}")
        finally:
            self.compare_button.config(state='normal')


    # --- Funciones para Visualización de Pérdidas ---

    def select_loss_file(self):
        loss_file = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")],
            title="Seleccionar Archivo CSV de Pérdidas"
        )
        if loss_file:
            self.loss_file_entry.delete(0, tk.END)
            self.loss_file_entry.insert(0, loss_file)
            self.plot_button.config(state='normal')

    def select_save_directory(self):
        save_dir = filedialog.askdirectory(title="Seleccionar Directorio de Guardado")
        if save_dir:
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.insert(0, save_dir)

    def toggle_save_options(self, *args):
        if self.save_png_var.get() or self.save_svg_var.get():
            self.save_dir_entry.config(state="normal")
            self.save_dir_button.config(state="normal")
        else:
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.config(state="disabled")
            self.save_dir_button.config(state="disabled")

    def plot_losses(self):
        loss_file = self.loss_file_entry.get()
        if not os.path.isfile(loss_file):
            messagebox.showerror("Error", "Seleccione un archivo CSV válido.")
            return

        save_png = self.save_png_var.get()
        save_svg = self.save_svg_var.get()
        save_dir = self.save_dir_entry.get()

        if (save_png or save_svg) and not os.path.isdir(save_dir):
            messagebox.showerror("Error", "Seleccione un directorio de guardado válido.")
            return

        try:
            # Leer el archivo CSV
            losses_df = pd.read_csv(loss_file)

            # Verificar que contenga las columnas necesarias
            if not {'Epoch', 'Train Loss', 'Val Loss'}.issubset(losses_df.columns):
                messagebox.showerror("Error", "El archivo CSV no contiene las columnas necesarias.")
                return

            # Crear la figura
            plt.figure(figsize=(8, 6))
            plt.plot(losses_df['Epoch'], losses_df['Train Loss'], label='Pérdida de Entrenamiento')
            plt.plot(losses_df['Epoch'], losses_df['Val Loss'], label='Pérdida de Validación')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.title('Pérdida de Entrenamiento y Validación por Época')
            plt.legend()
            plt.grid(True)

            # Aplicar el layout
            plt.tight_layout()

            # Guardar el gráfico si se seleccionó
            if save_png or save_svg:
                base_filename = os.path.splitext(os.path.basename(loss_file))[0]
                if save_png:
                    png_path = os.path.join(save_dir, f"{base_filename}.png")
                    plt.savefig(png_path, format='png')
                    self.logger.info(f"Gráfico guardado como {png_path}")
                if save_svg:
                    svg_path = os.path.join(save_dir, f"{base_filename}.svg")
                    plt.savefig(svg_path, format='svg')
                    self.logger.info(f"Gráfico guardado como {svg_path}")
                messagebox.showinfo("Éxito", "Gráfico guardado exitosamente.")
            else:
                self.logger.info("El gráfico no se guardó.")

            # Mostrar el gráfico después de guardarlo
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Se produjo un error al generar el gráfico: {e}")
            self.logger.error(f"Error al generar el gráfico: {e}")

    # --- Funciones para Métricas ---

    def calculate_metrics(self):
        iterations_str = self.iterations_entry.get()
        try:
            iterations = int(iterations_str)
            if iterations <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número de iteraciones válido (entero positivo).")
            return

        self.metrics_status_label.config(text="Calculando métricas...")
        self.calculate_metrics_button.config(state='disabled')
        self.metrics_tree.delete(*self.metrics_tree.get_children())

        # Iniciar el cálculo en un hilo separado
        threading.Thread(target=self.calculate_metrics_thread, args=(iterations,), daemon=True).start()
        
    def calculate_metrics_thread(self, iterations):
        try:
            # Listas para almacenar los valores reales y las predicciones
            real_soc_list = []
            pred_soc_combined_list = []
            pred_soc_f_list = []

            for i in range(iterations):
                # Definir los grupos objetivo
                grupos_objetivo = ['35', '36', '37', '38']

                # Verificar qué grupos están disponibles
                grupos_disponibles = [g for g in grupos_objetivo if g in self.dic_lista_dfs]
                if not grupos_disponibles:
                    self.logger.warning(f"Ninguno de los grupos {grupos_objetivo} está disponible en los datos.")
                    continue

                # Seleccionar un grupo aleatorio de los disponibles
                group = random.choice(grupos_disponibles)
                self.logger.info(f"Iteración {i+1}: Grupo seleccionado aleatoriamente: {group}")

                # Obtener datos del primer ciclo
                if 0 in self.dic_lista_dfs[group]:
                    df_first_cycle = self.dic_lista_dfs[group][0]
                else:
                    # Si el ciclo 0 no está disponible, usar el ciclo mínimo disponible
                    available_cycles = list(self.dic_lista_dfs[group].keys())
                    min_cycle = min(available_cycles)
                    df_first_cycle = self.dic_lista_dfs[group][min_cycle]
                    self.logger.warning(f"Iteración {i+1}: El ciclo 0 no está disponible en el grupo {group}. Usando ciclo {min_cycle}.")

                # Definir rangos basados en los cuantiles de 'Charge_Capacity(Ah)' del primer ciclo
                charge_capacities = df_first_cycle['Charge_Capacity(Ah)']
                low_quantile = charge_capacities.quantile(1/3)
                high_quantile = charge_capacities.quantile(2/3)
                ranges = {
                    'A': (charge_capacities.min(), low_quantile),
                    'B': (low_quantile, high_quantile),
                    'C': (high_quantile, charge_capacities.max())
                }
                self.logger.info(f"Iteración {i+1}: Rangos establecidos basados en 'Charge_Capacity(Ah)': {ranges}")

                # Seleccionar un ciclo aleatorio entre 0 y 100
                cycle = random.randint(0, 100)
                self.logger.info(f"Iteración {i+1}: Ciclo seleccionado aleatoriamente: {cycle}")

                # Verificar si el ciclo existe en el grupo seleccionado
                if cycle not in self.dic_lista_dfs[group]:
                    self.logger.warning(f"Iteración {i+1}: El ciclo {cycle} no existe en el grupo {group}.")
                    continue

                df_cycle = self.dic_lista_dfs[group][cycle]

                # Seleccionar un valor aleatorio de Voltage(V)
                V_test = random.choice(df_cycle['Voltage(V)'].tolist())
                charge_capacity_test = df_cycle.loc[df_cycle['Voltage(V)'] == V_test, 'Charge_Capacity(Ah)'].iloc[0]
                start_range = df_cycle[df_cycle['Charge_Capacity(Ah)'] == charge_capacity_test].index[0]
                self.logger.info(f"Iteración {i+1}: V_test seleccionado: {V_test}, Charge_Capacity_test: {charge_capacity_test}, start_range: {start_range}")

                # Determinar en qué rango está charge_capacity_test
                rango_de_Charge = None
                for rango, (low, high) in ranges.items():
                    if rango != 'C' and low <= charge_capacity_test < high:
                        rango_de_Charge = rango
                        break
                    elif rango == 'C' and high <= charge_capacity_test <= charge_capacities.max():
                        rango_de_Charge = rango
                        break

                if not rango_de_Charge:
                    # Asignar al rango más cercano
                    rango_de_Charge = self.find_closest_range(charge_capacity_test, ranges)
                    self.logger.warning(f"Iteración {i+1}: Charge_Capacity_test {charge_capacity_test} no cae en ninguno de los rangos A, B, C. Asignado al rango más cercano: {rango_de_Charge}")

                self.logger.info(f"Iteración {i+1}: Charge_Capacity_test {charge_capacity_test} cae en el rango {rango_de_Charge}")

                # Obtener el modelo correspondiente al rango_de_Charge
                with self.loaded_models_lock:
                    model_info = self.loaded_models.get(rango_de_Charge)
                if not model_info:
                    self.logger.warning(f"Iteración {i+1}: No se encontró el modelo para el segmento {rango_de_Charge}.")
                    continue

                model_selected = model_info['model']
                numero_de_datos = model_info['numero_de_datos']
                self.logger.info(f"Iteración {i+1}: Usando modelo {rango_de_Charge} con numero_de_datos: {numero_de_datos}")

                # Verificación de que el modelo seleccionado corresponde al segmento
                if not self.verify_model_segment(rango_de_Charge, model_selected):
                    self.logger.warning(f"Iteración {i+1}: El modelo seleccionado para el segmento '{rango_de_Charge}' no es el correcto.")
                    continue

                # Verificar si start_range es mayor que numero_de_datos
                if start_range <= numero_de_datos:
                    self.logger.warning(f"Iteración {i+1}: start_range ({start_range}) no es mayor que numero_de_datos ({numero_de_datos}). Seleccionando otro V_test.")
                    # Intentar seleccionar otro V_test hasta cumplir la condición
                    ciclos_disponibles = df_cycle[df_cycle.index > numero_de_datos]
                    if ciclos_disponibles.empty:
                        self.logger.warning(f"Iteración {i+1}: No se pudo encontrar un V_test que cumpla con la condición start_range > numero_de_datos.")
                        continue
                    V_test = random.choice(ciclos_disponibles['Voltage(V)'].tolist())
                    charge_capacity_test = df_cycle.loc[df_cycle['Voltage(V)'] == V_test, 'Charge_Capacity(Ah)'].iloc[0]
                    start_range = df_cycle[df_cycle['Charge_Capacity(Ah)'] == charge_capacity_test].index[0]
                    self.logger.info(f"Iteración {i+1}: Nuevo V_test seleccionado: {V_test}, Charge_Capacity_test: {charge_capacity_test}, start_range: {start_range}")

                    # Determinar nuevamente el rango de 'Charge_Capacity(Ah)'
                    rango_de_Charge = None
                    for rango, (low, high) in ranges.items():
                        if rango != 'C' and low <= charge_capacity_test < high:
                            rango_de_Charge = rango
                            break
                        elif rango == 'C' and high <= charge_capacity_test <= charge_capacities.max():
                            rango_de_Charge = rango
                            break

                    if not rango_de_Charge:
                        # Asignar al rango más cercano
                        rango_de_Charge = self.find_closest_range(charge_capacity_test, ranges)
                        self.logger.warning(f"Iteración {i+1}: Charge_Capacity_test {charge_capacity_test} no cae en ninguno de los rangos A, B, C. Asignado al rango más cercano: {rango_de_Charge}")

                    self.logger.info(f"Iteración {i+1}: Charge_Capacity_test {charge_capacity_test} cae en el rango {rango_de_Charge}")

                    # Obtener el modelo correspondiente al rango_de_Charge
                    with self.loaded_models_lock:
                        model_info = self.loaded_models.get(rango_de_Charge)
                    if not model_info:
                        self.logger.warning(f"Iteración {i+1}: No se encontró el modelo para el segmento {rango_de_Charge}.")
                        continue

                    model_selected = model_info['model']
                    numero_de_datos = model_info['numero_de_datos']
                    self.logger.info(f"Iteración {i+1}: Usando modelo {rango_de_Charge} con numero_de_datos: {numero_de_datos}")

                    # Verificación de que el modelo seleccionado corresponde al segmento
                    if not self.verify_model_segment(rango_de_Charge, model_selected):
                        self.logger.warning(f"Iteración {i+1}: El modelo seleccionado para el segmento '{rango_de_Charge}' no es el correcto.")
                        continue

                # Definir 'end_value' como 'charge_capacity_test'
                end_value = charge_capacity_test

                end_range = start_range
                start_range = end_range - numero_de_datos
                start_range = max(start_range, 0)  # Evitar índices negativos
                self.logger.info(f"Iteración {i+1}: start_range ajustado: {start_range}, end_range: {end_range}")

                # Construir DataFrame para el rango seleccionado
                df_selected = df_cycle.iloc[start_range:end_range].copy()

                # Extraer los vectores normalizados (ya están normalizados)
                V_vector_selected_normalized = df_selected['Voltage(V)'].values
                Current_vector_selected_normalized = df_selected['Current(A)'].values
                dVdQ_vector_selected_normalized = df_selected['dV/dQ'].values

                # Convertir a tensores y mover al dispositivo
                try:
                    X_selected = torch.tensor([
                        V_vector_selected_normalized,
                        Current_vector_selected_normalized,
                        dVdQ_vector_selected_normalized
                    ], dtype=torch.float32).unsqueeze(0).to(self.device)
                    self.logger.info(f"Iteración {i+1}: Tensor X_selected creado exitosamente.")
                except Exception as e:
                    self.logger.error(f"Iteración {i+1}: Error al convertir los datos a tensores para el modelo {rango_de_Charge}: {e}")
                    continue

                # Obtener el modelo F
                with self.loaded_models_lock:
                    model_f_info = self.loaded_models.get('F')
                if not model_f_info:
                    self.logger.warning(f"Iteración {i+1}: No se encontró el modelo para el segmento F.")
                    continue

                model_f = model_f_info['model']
                numero_de_datos_f = model_f_info['numero_de_datos']
                self.logger.info(f"Iteración {i+1}: Usando modelo F con numero_de_datos: {numero_de_datos_f}")

                # Construir DataFrame para el modelo F
                df_f = df_cycle.iloc[start_range:end_range].copy()

                # Extraer los vectores normalizados (ya están normalizados)
                V_vector_f_normalized = df_f['Voltage(V)'].values
                Current_vector_f_normalized = df_f['Current(A)'].values
                dVdQ_vector_f_normalized = df_f['dV/dQ'].values

                # Convertir a tensores para el modelo F y mover al dispositivo
                try:
                    X_f = torch.tensor([
                        V_vector_f_normalized,
                        Current_vector_f_normalized,
                        dVdQ_vector_f_normalized
                    ], dtype=torch.float32).unsqueeze(0).to(self.device)
                    self.logger.info(f"Iteración {i+1}: Tensor X_f creado exitosamente.")
                except Exception as e:
                    self.logger.error(f"Iteración {i+1}: Error al convertir los datos a tensores para el modelo F: {e}")
                    continue

                # Predicción del modelo seleccionado (normalizada)
                with torch.no_grad():
                    prediction_selected_normalized = model_selected(X_selected).item()

                self.logger.info(f"Iteración {i+1}: SoC predicha por el modelo {rango_de_Charge}: {prediction_selected_normalized}")

                # Predicción del modelo F (normalizada)
                with torch.no_grad():
                    prediction_f_normalized = model_f(X_f).item()

                self.logger.info(f"Iteración {i+1}: SoC predicha por el modelo F: {prediction_f_normalized}")

                # Almacenar los valores reales y predicciones
                real_soc_list.append(end_value)
                pred_soc_combined_list.append(prediction_selected_normalized)
                pred_soc_f_list.append(prediction_f_normalized)

            # Calcular RMSE para ambos métodos
            if real_soc_list:
                rmse_combined = np.sqrt(np.mean((np.array(pred_soc_combined_list) - np.array(real_soc_list))**2))
                rmse_f = np.sqrt(np.mean((np.array(pred_soc_f_list) - np.array(real_soc_list))**2))

                # Mostrar los resultados en el Treeview de métricas
                self.metrics_tree.insert("", "end", values=("RMSE Combinado (A+B+C vs Real)", f"{rmse_combined:.4f}"))
                self.metrics_tree.insert("", "end", values=("RMSE Modelo F (F vs Real)", f"{rmse_f:.4f}"))

                self.logger.info(f"RMSE Combinado (A+B+C vs Real): {rmse_combined}")
                self.logger.info(f"RMSE Modelo F (F vs Real): {rmse_f}")

                self.metrics_status_label.config(text="Cálculo de métricas completado.")
            else:
                messagebox.showwarning("Advertencia", "No se realizaron estimaciones válidas para calcular métricas.")
                self.metrics_status_label.config(text="No se calcularon métricas.")

        except Exception as e:
            messagebox.showerror("Error", f"Se produjo un error al calcular las métricas: {e}")
            self.logger.error(f"Error al calcular las métricas: {e}")
        finally:
            self.calculate_metrics_button.config(state='normal')
