# scripts/gui/train_tab.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from scripts.models.model_training import entrenar_evaluar_modelos
from scripts.data.data_preparation import (
    cargar_diccionario_desde_hdf5,
    extract_subsets,
    preparar_datos,
    process_group
)

from scripts.models.custom_model_manager import CustomModelManager
from scripts.models.training_parameters import get_training_parameters
from scripts.gui.metrics_display import MetricsDisplay
from scripts.gui.progress_handler import ProgressHandler
import logging
import pandas as pd

class TrainTab:
    def __init__(self, parent, custom_model_params):
        self.parent = parent
        self.custom_model_params = custom_model_params
        self.dic_lista_dfs_train = {}
        self.estadisticas_globales = {}
        self.loaded_models = []
        self.model_names = ['Model 1', 'Model 2', 'Model 3']
        self.create_widgets()

        # Configuración de Logging
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger('TrainTab')
        self.logger.setLevel(logging.DEBUG)

        # Crear un manejador de archivos
        fh = logging.FileHandler('train_tab.log')
        fh.setLevel(logging.DEBUG)

        # Crear un formato para los logs
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Añadir el manejador al logger
        self.logger.addHandler(fh)

    def create_widgets(self):
        # Frame Principal
        main_frame = ttk.Frame(self.parent, padding=10)
        main_frame.pack(fill='both', expand=True)

        # --- Selección de Archivo .h5 ---
        file_frame = ttk.LabelFrame(main_frame, text="Seleccionar Archivo de Datos (.h5)", padding=10)
        file_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky='we')

        self.h5_label_train = ttk.Label(file_frame, text="Archivo .h5:")
        self.h5_label_train.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.h5_entry_train = ttk.Entry(file_frame, width=40)
        self.h5_entry_train.grid(row=0, column=1, padx=5, pady=5, sticky='we')

        self.h5_button_train = ttk.Button(
            file_frame, text="Seleccionar", command=self.select_h5_file_train
        )
        self.h5_button_train.grid(row=0, column=2, padx=5, pady=5)

        self.load_button_train = ttk.Button(
            file_frame, text="Cargar Datos", command=self.load_data_train
        )
        self.load_button_train.grid(row=0, column=3, padx=5, pady=5)

        # --- Selección de Modelos ---
        models_frame = ttk.LabelFrame(main_frame, text="Seleccionar Modelos", padding=10)
        models_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky='we')

        self.model_vars = []
        for idx, model_name in enumerate(self.model_names):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(models_frame, text=model_name, variable=var)
            chk.grid(row=0, column=idx, padx=5, pady=5, sticky='w')
            self.model_vars.append(var)

        self.view_params_button = ttk.Button(
            models_frame, text="Ver Parámetros", command=self.view_model_params
        )
        self.view_params_button.grid(row=0, column=len(self.model_names), padx=5, pady=5)

        self.add_custom_model_button = ttk.Button(
            models_frame, text="Agregar Modelo Personalizado", command=self.open_custom_model_window
        )
        self.add_custom_model_button.grid(row=0, column=len(self.model_names)+1, padx=5, pady=5)

        # --- Selección de Segmentos ---
        segments_frame = ttk.LabelFrame(main_frame, text="Seleccionar Segmentos", padding=10)
        segments_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky='we')

        self.segment_vars = []
        self.segment_names = ['A', 'B', 'C', 'F']
        for idx, segment_name in enumerate(self.segment_names):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(segments_frame, text=segment_name, variable=var)
            chk.grid(row=0, column=idx, padx=5, pady=5, sticky='w')
            self.segment_vars.append(var)

        # --- Selección de Nº de Datos ---
        data_num_frame = ttk.LabelFrame(main_frame, text="Seleccionar Nº de Datos", padding=10)
        data_num_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky='we')

        self.n_data_vars = []
        self.n_data_options = [20, 40, 55, 60]
        for idx, n_data in enumerate(self.n_data_options):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(data_num_frame, text=str(n_data), variable=var)
            chk.grid(row=0, column=idx, padx=5, pady=5, sticky='w')
            self.n_data_vars.append(var)
            # Seleccionar 55 por defecto
            if n_data == 55:
                var.set(True)

        # --- Parámetros de Entrenamiento ---
        params_frame = ttk.LabelFrame(main_frame, text="Parámetros de Entrenamiento", padding=10)
        params_frame.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky='we')

        # Nº de Iteraciones
        self.n_iter_label = ttk.Label(params_frame, text="Nº de Iteraciones:")
        self.n_iter_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.n_iter_entry = ttk.Entry(params_frame, width=10)
        self.n_iter_entry.insert(0, "50")
        self.n_iter_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        # Corte
        self.corte_label = ttk.Label(params_frame, text="Corte:")
        self.corte_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.corte_entry = ttk.Entry(params_frame, width=10)
        self.corte_entry.insert(0, "100")
        self.corte_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Épocas
        self.epochs_label = ttk.Label(params_frame, text="Épocas:")
        self.epochs_label.grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.epochs_entry = ttk.Entry(params_frame, width=10)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=1, column=3, padx=5, pady=5, sticky='w')

        # --- Carpeta para Guardar Modelos ---
        save_model_frame = ttk.LabelFrame(main_frame, text="Guardar Modelos", padding=10)
        save_model_frame.grid(row=5, column=0, columnspan=4, padx=5, pady=5, sticky='we')

        self.save_model_label = ttk.Label(save_model_frame, text="Carpeta para Guardar:")
        self.save_model_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.save_model_entry = ttk.Entry(save_model_frame, width=40)
        self.save_model_entry.grid(row=0, column=1, padx=5, pady=5, sticky='we')

        self.save_model_button = ttk.Button(
            save_model_frame, text="Seleccionar", command=self.select_save_model_folder
        )
        self.save_model_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Botones de Acción ---
        action_frame = ttk.Frame(main_frame, padding=5)
        action_frame.grid(row=6, column=0, columnspan=4, pady=10)

        self.train_button = ttk.Button(
            action_frame, text="Entrenar Modelos", command=self.train_models, width=20, state='disabled'
        )
        self.train_button.pack(pady=5)

        # Añadir el botón de generar ejemplos (si aplica)
        self.generate_examples_button = ttk.Button(
            action_frame, text="Generar Ejemplos", command=self.generate_examples, width=20
        )
        self.generate_examples_button.pack(pady=5)

        # --- Barra de Progreso ---
        progress_frame = ttk.Frame(main_frame, padding=5)
        progress_frame.grid(row=7, column=0, columnspan=4, pady=5, sticky='we')

        self.progress = ttk.Progressbar(progress_frame, orient='horizontal', mode='indeterminate')
        self.progress.pack(fill='x', padx=5, pady=5)

        # Etiqueta para mostrar mensajes de progreso
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(pady=(0,5))

        # --- Cuadro de Texto para Métricas ---
        metrics_frame = ttk.LabelFrame(main_frame, text="Métricas de Entrenamiento", padding=10)
        metrics_frame.grid(row=8, column=0, columnspan=4, padx=5, pady=5, sticky='we')

        # Iniciar MetricsDisplay
        self.metrics_display = MetricsDisplay(metrics_frame)

        # --- Etiqueta de Estado ---
        self.status_label3 = ttk.Label(main_frame, text="")
        self.status_label3.grid(row=9, column=0, columnspan=4, padx=5, pady=5, sticky='w')

        # Iniciar ProgressHandler
        self.progress_handler = ProgressHandler(self.parent, self.progress_label)

    def select_h5_file_train(self):
        h5_path = filedialog.askopenfilename(
            filetypes=[("Archivo HDF5", "*.h5")],
            title="Seleccionar Archivo de Entrenamiento"
        )
        if h5_path:
            self.h5_entry_train.delete(0, tk.END)
            self.h5_entry_train.insert(0, h5_path)

    def load_data_train(self):
        h5_path = self.h5_entry_train.get()
        if not os.path.isfile(h5_path):
            messagebox.showerror("Error", "Seleccione un archivo .h5 válido.")
            return
        self.status_label3.config(text="Cargando datos...")
        self.train_button.state(['disabled'])
        # Limpiar las métricas previas
        self.metrics_display.metrics_tree.delete(*self.metrics_display.metrics_tree.get_children())
        self.progress_label.config(text="")
        threading.Thread(target=self.load_data_train_thread, args=(h5_path,), daemon=True).start()

    def load_data_train_thread(self, h5_path):
        try:
            self.logger.info(f"Cargando datos desde {h5_path}")
            self.dic_lista_dfs_train, self.estadisticas_globales = cargar_diccionario_desde_hdf5(h5_path)
            self.logger.info("Datos cargados y normalizados con éxito.")
            self.status_label3.config(text="Datos cargados y normalizados con éxito.")
            # Habilitar el botón de entrenar modelos
            self.train_button.state(['!disabled'])
        except Exception as e:
            error_msg = f"Se produjo un error al cargar los datos: {e}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.status_label3.config(text="")
        finally:
            self.train_button.state(['!disabled'])

    def view_model_params(self):
        selected_models = [name for var, name in zip(
            self.model_vars, self.model_names) if var.get()]
        if len(selected_models) != 1:
            messagebox.showerror("Error", "Seleccione un único modelo para ver sus parámetros.")
            return
        model_name = selected_models[0]
        params_text = self.get_model_params_text(model_name)
        messagebox.showinfo(f"Parámetros del {model_name}", params_text)

    def get_model_params_text(self, model_name):
        if model_name == 'Model 1':
            params = """
Conv. Filters: [32, 64, 128]
Conv. Kernel Sizes: [3, 3, 3]
FC Units: 256
"""
        elif model_name == 'Model 2':
            params = """
Conv. Filters: [44, 42, 54]
Conv. Kernel Sizes: [7, 4, 16]
FC Units: 60
"""
        elif model_name == 'Model 3':
            params = """
Conv. Filters: [11, 60, 56]
Conv. Kernel Sizes: [10, 1, 14]
FC Units: 60
"""
        elif model_name.startswith('Modelo Personalizado'):
            params_dict = self.custom_model_params.get(model_name, {})
            if params_dict:
                params_list = [f"{k}: {v}" for k, v in params_dict.items()]
                params = '\n'.join(params_list)
            else:
                params = "No hay parámetros disponibles para este modelo."
        else:
            params = "Modelo no reconocido."
        return params

    def open_custom_model_window(self):
        self.custom_model_manager = CustomModelManager(
            self.parent, self.model_names, self.custom_model_params
        )
        self.custom_model_manager.add_custom_model()

        # Actualizar los checkboxes de modelos si se agregó un nuevo modelo personalizado
        self.update_model_checkboxes()

    def update_model_checkboxes(self):
        # Limpiar los checkboxes existentes
        models_frame = self.parent.children['!labelframe']
        for widget in models_frame.winfo_children():
            widget.destroy()

        self.model_vars = []
        for idx, model_name in enumerate(self.model_names):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(models_frame, text=model_name, variable=var)
            chk.grid(row=0, column=idx, padx=5, pady=5, sticky='w')
            self.model_vars.append(var)

        self.view_params_button = ttk.Button(
            models_frame, text="Ver Parámetros", command=self.view_model_params
        )
        self.view_params_button.grid(row=0, column=len(self.model_names), padx=5, pady=5)

        self.add_custom_model_button = ttk.Button(
            models_frame, text="Agregar Modelo Personalizado", command=self.open_custom_model_window
        )
        self.add_custom_model_button.grid(row=0, column=len(self.model_names)+1, padx=5, pady=5)

    def select_save_model_folder(self):
        save_folder = filedialog.askdirectory(title="Seleccionar Carpeta para Guardar Modelos")
        if save_folder:
            self.save_model_entry.delete(0, tk.END)
            self.save_model_entry.insert(0, save_folder)

    def train_models(self):
        modelos_seleccionados = [name for var, name in zip(
            self.model_vars, self.model_names) if var.get()]
        segmentos_seleccionados = [name for var, name in zip(
            self.segment_vars, self.segment_names) if var.get()]
        numeros_de_datos_seleccionados = [n_data for var, n_data in zip(
            self.n_data_vars, self.n_data_options) if var.get()]
        if not modelos_seleccionados:
            messagebox.showerror("Error", "Seleccione al menos un modelo.")
            return
        if not segmentos_seleccionados:
            messagebox.showerror(
                "Error", "Seleccione al menos un segmento.")
            return
        if not numeros_de_datos_seleccionados:
            messagebox.showerror(
                "Error", "Seleccione al menos un número de datos.")
            return

        parametros = get_training_parameters(
            self.n_iter_entry, numeros_de_datos_seleccionados, self.corte_entry, self.epochs_entry)
        if parametros is None:
            return  # Error ya mostrado en get_training_parameters

        # Verificar carpeta de guardado
        save_folder = self.save_model_entry.get()
        if not os.path.isdir(save_folder):
            messagebox.showerror("Error", "Seleccione una carpeta válida para guardar los modelos.")
            return

        self.status_label3.config(text="Entrenando modelos...")
        self.train_button.state(['disabled'])
        self.progress.start()
        self.progress_label.config(text="Iniciando entrenamiento...")
        # Limpiar las métricas previas
        self.metrics_display.metrics_tree.delete(*self.metrics_display.metrics_tree.get_children())
        threading.Thread(target=self.train_models_thread, args=(
            modelos_seleccionados, segmentos_seleccionados, parametros), daemon=True).start()

    def train_models_thread(self, modelos_seleccionados, segmentos_seleccionados, parametros):
        try:
            # Ejecutar el entrenamiento y evaluación de modelos
            df_metricas = entrenar_evaluar_modelos(
                (self.dic_lista_dfs_train, self.estadisticas_globales),
                modelos_seleccionados,
                segmentos_seleccionados,
                parametros,
                save_folder=self.save_model_entry.get(),
                custom_model_params=self.custom_model_params,
                progress_callback=self.progress_handler.progress_callback
            )

            # Actualizar la tabla con las métricas
            self.parent.after(0, self.metrics_display.display_metrics, df_metricas)

            self.status_label3.config(text="Modelos entrenados con éxito.")
            self.logger.info("Entrenamiento completado con éxito.")
            messagebox.showinfo("Éxito", "Los modelos han sido entrenados y guardados exitosamente.")
        except Exception as e:
            error_msg = f"Se produjo un error durante el entrenamiento: {e}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.status_label3.config(text="")
        finally:
            self.parent.after(0, self.progress.stop)
            self.parent.after(0, lambda: self.train_button.state(['!disabled']))

    # --- Nuevas Funciones para Generar Ejemplos ---

    def generate_examples(self):
        h5_path = self.h5_entry_train.get()
        if not os.path.isfile(h5_path):
            messagebox.showerror("Error", "Seleccione un archivo .h5 válido antes de generar ejemplos.")
            return

        # Verificar que los datos hayan sido cargados
        if not self.dic_lista_dfs_train:
            messagebox.showerror("Error", "Cargue los datos antes de generar ejemplos.")
            return

        self.status_label3.config(text="Generando ejemplos...")
        self.generate_examples_button.state(['disabled'])

        # Iniciar la generación de ejemplos en un hilo separado
        threading.Thread(target=self.generate_examples_thread, daemon=True).start()

    def generate_examples_thread(self):
        try:
            # Obtener el primer número de datos seleccionado
            numeros_de_datos_seleccionados = [n_data for var, n_data in zip(
                self.n_data_vars, self.n_data_options) if var.get()]
            if numeros_de_datos_seleccionados:
                numero_de_datos = numeros_de_datos_seleccionados[0]
            else:
                messagebox.showerror("Error", "Seleccione al menos un número de datos para generar ejemplos.")
                self.status_label3.config(text="")
                self.parent.after(0, lambda: self.generate_examples_button.state(['!disabled']))
                return

            # Seleccionar un par de grupos y ciclos para generar ejemplos
            ejemplos_subsets = []
            ejemplos_end_values = []

            grupos_seleccionados = list(self.dic_lista_dfs_train.items())[:2]  # Tomar dos grupos
            for group_id, group_dict in grupos_seleccionados:
                ciclos_seleccionados = list(group_dict.items())[:1]  # Tomar un ciclo por grupo
                for cycle_index, df in ciclos_seleccionados:
                    # Eliminar la normalización redundante
                    subsets, end_values = extract_subsets(
                        {cycle_index: df},  # Pasar el DataFrame ya normalizado
                        self.estadisticas_globales,
                        numero_de_datos=numero_de_datos,  # Usar el valor seleccionado
                        corte=int(self.corte_entry.get()),          # Usar el valor del campo de entrada
                        part=1,                                      # Ajustar según tu criterio
                        columnas=['Voltage(V)', 'Current(A)', 'dV/dQ']
                    )
                    if subsets and end_values:
                        ejemplos_subsets.append(subsets[0])  # Tomar el primer subset
                        ejemplos_end_values.append(end_values[0])  # Tomar el primer end_value

            if not ejemplos_subsets or not ejemplos_end_values:
                messagebox.showwarning("Advertencia", "No se pudieron generar ejemplos con los parámetros actuales.")
                self.status_label3.config(text="No se generaron ejemplos.")
                return

            # Convertir los ejemplos a DataFrames para una mejor visualización
            # Añadir una columna 'Ejemplo' para identificar cada conjunto
            ejemplos = []
            for i, (subset, end_value) in enumerate(zip(ejemplos_subsets, ejemplos_end_values), start=1):
                df_subset = pd.DataFrame(subset, columns=['Voltage(V)', 'Current(A)', 'dV/dQ'])
                df_subset['End_Value'] = end_value
                df_subset['Ejemplo'] = f'Ejemplo {i}'
                ejemplos.append(df_subset)

            ejemplos_df = pd.concat(ejemplos, ignore_index=True)

            # Mostrar los ejemplos en una nueva ventana
            self.parent.after(0, self.show_examples_window, ejemplos_df)

            self.logger.info("Ejemplos generados con éxito.")
            self.status_label3.config(text="Ejemplos generados con éxito.")
        except Exception as e:
            error_msg = f"Se produjo un error al generar ejemplos: {e}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
            self.status_label3.config(text="")
        finally:
            self.parent.after(0, lambda: self.generate_examples_button.state(['!disabled']))

    def show_examples_window(self, ejemplos_df):
        examples_window = tk.Toplevel(self.parent)
        examples_window.title("Ejemplos de Subsets y End_Values")
        examples_window.geometry("800x600")

        # Crear un Treeview para mostrar los ejemplos
        tree = ttk.Treeview(examples_window)
        tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Definir las columnas
        tree['columns'] = list(ejemplos_df.columns)
        tree['show'] = 'headings'

        for col in ejemplos_df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='center')

        # Insertar los datos
        for _, row in ejemplos_df.iterrows():
            tree.insert('', tk.END, values=list(row))

        # Agregar una barra de desplazamiento
        scrollbar = ttk.Scrollbar(examples_window, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
