# scripts/gui/visualize.py


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import logging
from scripts.data.data_visualization import (
    cargar_datos_h5,
    graficar_ciclos_separados,
    graficar_datos_lista_delta,
    mostrar_graficos
)
from scripts.data.data_preparation import find_subset_by_number_of_data
import os
from PIL import Image, ImageTk
import random  # Añadido para selección aleatoria de ciclo

# Configurar logger para o módulo
logger = logging.getLogger(__name__)

class VisualizeTab:
    def __init__(self, parent):
        self.parent = parent
        self.dic_lista_dfs = {}
        self.create_widgets()

    def create_widgets(self):
        # Configurar el grid para que los widgets se expandan adecuadamente
        self.parent.columnconfigure(1, weight=1)
        self.parent.columnconfigure(3, weight=1)

        # Archivo .h5
        self.h5_label = ttk.Label(self.parent, text="Archivo .h5:")
        self.h5_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.h5_entry = ttk.Entry(self.parent, width=50)
        self.h5_entry.grid(row=0, column=1, padx=5, pady=5, sticky='we')

        self.h5_button = ttk.Button(
            self.parent, text="Seleccionar", command=self.select_h5_file
        )
        self.h5_button.grid(row=0, column=2, padx=5, pady=5)

        # Botón para cargar datos
        self.load_button = ttk.Button(
            self.parent, text="Cargar Datos", command=self.load_data
        )
        self.load_button.grid(row=0, column=3, padx=5, pady=5)

        # Opción para seleccionar tipo de gráfico
        self.plot_type_label = ttk.Label(self.parent, text="Tipo de Gráfico:")
        self.plot_type_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

        self.plot_type_var = tk.StringVar(value="Single")
        self.single_radio = ttk.Radiobutton(
            self.parent, text="Único Ciclo", variable=self.plot_type_var, value="Single", command=self.update_plot_type
        )
        self.single_radio.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.multiple_radio = ttk.Radiobutton(
            self.parent, text="Rango de Ciclos", variable=self.plot_type_var, value="Multiple", command=self.update_plot_type
        )
        self.multiple_radio.grid(row=1, column=2, padx=5, pady=5, sticky='w')

        # Marco para parámetros de gráfico
        self.param_frame = ttk.LabelFrame(self.parent, text="Parámetros de Gráfico", padding=10)
        self.param_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=10, sticky='we')

        # Seleccionar grupo
        self.group_label = ttk.Label(self.param_frame, text="Grupo:")
        self.group_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.group_combobox = ttk.Combobox(self.param_frame, state="readonly")
        self.group_combobox.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.group_combobox.bind("<<ComboboxSelected>>", self.update_cycle_count)

        # Etiqueta para mostrar la cantidad de ciclos
        self.cycle_count_label = ttk.Label(self.param_frame, text="Cantidad de Ciclos: 0")
        self.cycle_count_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        # Parámetros para un único ciclo
        self.single_cycle_label = ttk.Label(self.param_frame, text="Ciclo:")
        self.single_cycle_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

        self.single_cycle_entry = ttk.Entry(self.param_frame, width=10)
        self.single_cycle_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Parámetros para rango de ciclos
        self.range_label = ttk.Label(self.param_frame, text="Ciclo Inicial:")
        self.range_label.grid(row=2, column=0, padx=5, pady=5, sticky='e')

        self.range_start_entry = ttk.Entry(self.param_frame, width=10)
        self.range_start_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        self.range_end_label = ttk.Label(self.param_frame, text="Ciclo Final:")
        self.range_end_label.grid(row=3, column=0, padx=5, pady=5, sticky='e')

        self.range_end_entry = ttk.Entry(self.param_frame, width=10)
        self.range_end_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')

        # Parámetro Salto (solo para múltiples ciclos)
        self.salto_label = ttk.Label(self.param_frame, text="Salto:")
        self.salto_label.grid(row=4, column=0, padx=5, pady=5, sticky='e')

        self.salto_entry = ttk.Entry(self.param_frame, width=10)
        self.salto_entry.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        self.salto_entry.insert(0, "1")  # Valor predeterminado

        # Opción para guardar gráficos
        self.save_var = tk.BooleanVar()
        self.save_checkbutton = ttk.Checkbutton(
            self.param_frame, text="Guardar Gráficos", variable=self.save_var, command=self.toggle_save_options
        )
        self.save_checkbutton.grid(row=5, column=0, padx=5, pady=5, sticky='e')

        self.save_folder_label = ttk.Label(self.param_frame, text="Carpeta de Guardado:")
        self.save_folder_label.grid(row=5, column=1, padx=5, pady=5, sticky='e')

        self.save_folder_entry = ttk.Entry(self.param_frame, width=30, state="disabled")
        self.save_folder_entry.grid(row=5, column=2, padx=5, pady=5, sticky='w')

        self.save_folder_button = ttk.Button(
            self.param_frame, text="Seleccionar", command=self.select_save_folder, state="disabled"
        )
        self.save_folder_button.grid(row=5, column=3, padx=5, pady=5, sticky='w')

        # Opción para guardar como SVG
        self.svg_var = tk.BooleanVar()
        self.svg_checkbutton = ttk.Checkbutton(
            self.param_frame, text="Guardar como SVG", variable=self.svg_var, state="disabled"
        )
        self.svg_checkbutton.grid(row=6, column=0, padx=5, pady=5, sticky='e')

        # Botón para generar gráficos
        self.plot_button = ttk.Button(
            self.parent, text="Generar Gráficos", command=self.plot_data, state='disabled'
        )
        self.plot_button.grid(row=3, column=1, columnspan=2, padx=5, pady=20)

        # Opciones para visualizar los gráficos
        self.display_var = tk.BooleanVar()
        self.display_checkbutton = ttk.Checkbutton(
            self.param_frame, text="Mostrar Gráficos en la Aplicación", variable=self.display_var
        )
        self.display_checkbutton.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        # Etiqueta de estado
        self.status_label2 = ttk.Label(self.parent, text="")
        self.status_label2.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky='w')

        # --- Nuevos Elementos para Extraer y Graficar Subconjunto ---
        self.extract_frame = ttk.LabelFrame(self.parent, text="Extraer y Graficar Subconjunto", padding=10)
        self.extract_frame.grid(row=5, column=0, columnspan=4, padx=5, pady=10, sticky='we')

        # Entrada para Número de Datos
        self.num_data_label = ttk.Label(self.extract_frame, text="Número de Datos:")
        self.num_data_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.num_data_entry = ttk.Entry(self.extract_frame, width=10)
        self.num_data_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.num_data_entry.insert(0, "10")  # Valor predeterminado

        # Opción para guardar el subconjunto como SVG
        self.extract_svg_var = tk.BooleanVar()
        self.extract_svg_checkbutton = ttk.Checkbutton(
            self.extract_frame, text="Guardar Gráfico como SVG", variable=self.extract_svg_var
        )
        self.extract_svg_checkbutton.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        # Botón para extraer y graficar
        self.extract_plot_button = ttk.Button(
            self.extract_frame, text="Extraer y Graficar Subconjunto", command=self.extract_and_plot_subset
        )
        self.extract_plot_button.grid(row=0, column=3, padx=5, pady=5)

        # --- Fin de Nuevos Elementos ---

        # Configuración inicial de la interfaz
        self.update_plot_type()

    def select_h5_file(self):
        h5_path = filedialog.askopenfilename(
            filetypes=[("Archivo HDF5", "*.h5")]
        )
        if h5_path:
            self.h5_entry.delete(0, tk.END)
            self.h5_entry.insert(0, h5_path)
            logger.info(f"Archivo .h5 seleccionado: {h5_path}")

    def load_data(self):
        h5_path = self.h5_entry.get()
        if not os.path.isfile(h5_path):
            messagebox.showerror("Error", "Seleccione un archivo .h5 válido.")
            logger.error("Archivo .h5 inválido o no encontrado.")
            return

        self.status_label2.config(text="Cargando datos...")
        self.plot_button.config(state='disabled')
        threading.Thread(target=self.load_data_thread, args=(h5_path,), daemon=True).start()

    def load_data_thread(self, h5_path):
        try:
            logger.info(f"Iniciando carga de datos desde: {h5_path}")
            self.dic_lista_dfs = cargar_datos_h5(h5_path)
            grupos = sorted(list(self.dic_lista_dfs.keys()))
            self.group_combobox.config(values=grupos, state="readonly")
            if grupos:
                self.group_combobox.current(0)
                self.parent.after(0, self.update_cycle_count)
            self.parent.after(0, lambda: self.plot_button.config(state='normal'))
            self.parent.after(0, lambda: self.status_label2.config(text="Datos cargados con éxito."))
            logger.info("Datos cargados con éxito.")
        except Exception as e:
            self.parent.after(0, lambda: messagebox.showerror("Error", f"Se produjo un error: {e}"))
            logger.error(f"Error al cargar datos: {e}")
            self.parent.after(0, lambda: self.status_label2.config(text=""))

    def select_save_folder(self):
        save_folder = filedialog.askdirectory()
        if save_folder:
            self.save_folder_entry.delete(0, tk.END)
            self.save_folder_entry.insert(0, save_folder)
            logger.info(f"Carpeta seleccionada para guardar gráficos: {save_folder}")

    def toggle_save_options(self):
        if self.save_var.get():
            self.save_folder_entry.config(state="normal")
            self.save_folder_button.config(state="normal")
            self.svg_checkbutton.config(state="normal")
            logger.info("Opciones de guardado habilitadas.")
        else:
            self.save_folder_entry.delete(0, tk.END)
            self.save_folder_entry.config(state="disabled")
            self.save_folder_button.config(state="disabled")
            self.svg_var.set(False)
            self.svg_checkbutton.config(state="disabled")
            logger.info("Opciones de guardado deshabilitadas.")

    def update_plot_type(self):
        plot_type = self.plot_type_var.get()
        if plot_type == "Single":
            # Habilitar campos para um único ciclo e desabilitar os de múltiplos ciclos
            self.single_cycle_label.config(state="normal")
            self.single_cycle_entry.config(state="normal")
            self.range_label.config(state="disabled")
            self.range_start_entry.config(state="disabled")
            self.range_end_label.config(state="disabled")
            self.range_end_entry.config(state="disabled")
            self.salto_label.config(state="disabled")
            self.salto_entry.config(state="disabled")
        elif plot_type == "Multiple":
            # Desabilitar campo único e habilitar campos de múltiplos ciclos
            self.single_cycle_label.config(state="disabled")
            self.single_cycle_entry.config(state="disabled")
            self.range_label.config(state="normal")
            self.range_start_entry.config(state="normal")
            self.range_end_label.config(state="normal")
            self.range_end_entry.config(state="normal")
            self.salto_label.config(state="normal")
            self.salto_entry.config(state="normal")

    def update_cycle_count(self, event=None):
        grupo = self.group_combobox.get()
        if grupo in self.dic_lista_dfs:
            cantidad = len(self.dic_lista_dfs[grupo])
            self.cycle_count_label.config(text=f"Cantidad de Ciclos: {cantidad}")
            logger.info(f"Grupo '{grupo}' seleccionado con {cantidad} ciclos disponibles.")
        else:
            self.cycle_count_label.config(text="Cantidad de Ciclos: 0")
            logger.warning(f"Grupo '{grupo}' no contiene ciclos.")

    def plot_data(self):
        # Reset previous plot parameters
        self.update_plot_type()
        grupo = self.group_combobox.get()
        plot_type = self.plot_type_var.get()
        ruta_guardado = self.save_folder_entry.get() if self.save_var.get() else None
        guardar_svg = self.svg_var.get()

        # Inform logger about the graph generation starting point with current parameters
        logger.info(f"Iniciando generación de gráficos para grupo '{grupo}' en modo '{plot_type}'.")

        # Reset selected cycles to avoid retaining old data
        ciclos_seleccionados = []

        if plot_type == "Single":
            ciclo_texto = self.single_cycle_entry.get()
            if not ciclo_texto.isdigit():
                messagebox.showerror("Error", "Ingrese un número de ciclo válido.")
                logger.error("Número de ciclo inválido en modo 'Único Ciclo'.")
                return
            ciclos_seleccionados = [int(ciclo_texto)]
            salto = 1
        elif plot_type == "Multiple":
            inicio_texto = self.range_start_entry.get()
            fin_texto = self.range_end_entry.get()
            salto_texto = self.salto_entry.get()

            if not (inicio_texto.isdigit() and fin_texto.isdigit() and salto_texto.isdigit()):
                messagebox.showerror("Error", "Ingrese números de ciclo válidos para el rango y salto.")
                logger.error("Valores inválidos para rango o salto en modo 'Rango de Ciclos'.")
                return

            inicio_cycle = int(inicio_texto)
            fin_cycle = int(fin_texto)
            salto = int(salto_texto)

            if inicio_cycle > fin_cycle:
                messagebox.showerror("Error", "El ciclo inicial no puede ser mayor que el ciclo final.")
                logger.error("El ciclo inicial es mayor que el final.")
                return

            if salto < 1:
                messagebox.showerror("Error", "El salto debe ser un número entero positivo.")
                logger.error("Valor de salto inválido (menor que 1).")
                return

            # Generate new cycle range based on current inputs
            ciclos_seleccionados = list(range(inicio_cycle, fin_cycle + 1, salto))
        else:
            messagebox.showerror("Error", "Tipo de gráfico no reconocido.")
            logger.error("Tipo de gráfico no reconocido.")
            return

        if not ciclos_seleccionados:
            messagebox.showerror("Error", "No se seleccionaron ciclos para graficar.")
            logger.error("Ningún ciclo seleccionado para graficar.")
            return

        # Update UI with progress status and reset plot button state after threading is complete
        self.status_label2.config(text="Generando gráficos...")
        self.plot_button.config(state='disabled')
        threading.Thread(target=self.plot_data_thread, args=(grupo, ciclos_seleccionados, salto, ruta_guardado, guardar_svg), daemon=True).start()

    def plot_data_thread(self, grupo, ciclos_seleccionados, salto, ruta_guardado, guardar_svg):
        try:
            logger.info(f"Generando gráficos para el grupo '{grupo}' con ciclos {ciclos_seleccionados} y salto {salto}.")
            graficar_ciclos_separados(self.dic_lista_dfs, grupo, ciclos_seleccionados, salto, ruta_guardado, guardar_svg)
            self.parent.after(0, lambda: self.status_label2.config(text="Gráficos generados con éxito."))
            logger.info("Gráficos generados exitosamente.")
            if self.display_var.get():
                mostrar_graficos(grupo, ciclos_seleccionados, salto, ruta_guardado)
        except Exception as e:
            self.parent.after(0, lambda: messagebox.showerror("Error", f"Se produjo un error al generar los gráficos: {e}"))
            self.parent.after(0, lambda: self.status_label2.config(text=""))
            logger.error(f"Error al generar los gráficos: {e}")
        finally:
            self.parent.after(0, lambda: self.plot_button.config(state='normal'))

    def extract_and_plot_subset(self):
        # Reset parameters to avoid old values affecting the new extraction
        grupo = self.group_combobox.get()
        num_datos_texto = self.num_data_entry.get()

        if not num_datos_texto.isdigit():
            messagebox.showerror("Error", "Ingrese un número de datos válido.")
            logger.error("Número de datos inválido para extracción de subconjunto.")
            return

        num_datos = int(num_datos_texto)

        # Ensure the group is valid and contains cycles
        if grupo not in self.dic_lista_dfs:
            messagebox.showerror("Error", f"El grupo '{grupo}' no está disponible.")
            logger.error(f"Grupo '{grupo}' no está disponible para extracción.")
            return

        available_cycles = list(self.dic_lista_dfs[grupo].keys())
        if not available_cycles:
            messagebox.showerror("Error", f"No hay ciclos disponibles en el grupo '{grupo}'.")
            logger.error(f"No hay ciclos disponibles en el grupo '{grupo}'.")
            return

        # Select a random cycle index for new subset
        ciclo_index = random.choice(available_cycles)
        logger.info(f"Ciclo seleccionado aleatoriamente: {ciclo_index}")

        # Reset any existing data selection
        df = self.dic_lista_dfs[grupo][ciclo_index]
        start_idx, end_idx = find_subset_by_number_of_data(df, numero_de_datos=num_datos, part=0)
        
        if start_idx is None or end_idx is None:
            messagebox.showerror("Error", "No se pudo extraer un subconjunto válido de datos.")
            logger.error("Extracción de subconjunto fallida.")
            return

        # Update UI and start new thread for subset plotting
        self.status_label2.config(text="Generando gráfico del subconjunto...")
        self.extract_plot_button.config(state='disabled')
        threading.Thread(target=self.extract_and_plot_subset_thread, args=(grupo, ciclo_index, start_idx, end_idx), daemon=True).start()

    def extract_and_plot_subset_thread(self, grupo, ciclo_index, start_idx, end_idx):
        try:
            logger.info(f"Extrayendo subconjunto para grupo '{grupo}', ciclo {ciclo_index}, índices {start_idx}-{end_idx}.")
            ruta_guardado = self.save_folder_entry.get() if self.save_var.get() else None
            guardar_svg = self.extract_svg_var.get()
            graficar_datos_lista_delta(self.dic_lista_dfs, grupo, ciclo_index, start_idx, end_idx, ruta_guardado, guardar_svg)
            self.parent.after(0, lambda: self.status_label2.config(text="Gráfico del subconjunto generado con éxito."))
            logger.info("Gráfico del subconjunto generado exitosamente.")
        except Exception as e:
            self.parent.after(0, lambda: messagebox.showerror("Error", f"Se produjo un error al graficar el subconjunto: {e}"))
            self.parent.after(0, lambda: self.status_label2.config(text=""))
            logger.error(f"Error al graficar el subconjunto: {e}")
        finally:
            self.parent.after(0, lambda: self.extract_plot_button.config(state='normal'))
