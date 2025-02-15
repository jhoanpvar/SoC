# scripts/gui/process_tab.py

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import logging
from multiprocessing import Process
from scripts.data.data_processing import flujo_principal
import os

# Configurar logger para el módulo
logger = logging.getLogger(__name__)

class ProcessTab:
    def __init__(self, parent):
        self.parent = parent
        self.process = None  # Referencia al proceso de procesamiento
        self.create_widgets()

    def create_widgets(self):
        # Configurar el grid para que los widgets se expandan adecuadamente
        self.parent.columnconfigure(1, weight=1)

        # Archivo ZIP
        self.zip_label = ttk.Label(self.parent, text="Archivo ZIP:")
        self.zip_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.zip_entry = ttk.Entry(self.parent, width=50)
        self.zip_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        self.zip_button = ttk.Button(
            self.parent, text="Seleccionar", command=self.select_zip)
        self.zip_button.grid(row=0, column=2, padx=5, pady=5)

        # Carpeta de salida
        self.output_label = ttk.Label(self.parent, text="Carpeta de salida:")
        self.output_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.output_entry = ttk.Entry(self.parent, width=50)
        self.output_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        self.output_button = ttk.Button(
            self.parent, text="Seleccionar", command=self.select_output_folder)
        self.output_button.grid(row=1, column=2, padx=5, pady=5)

        # Botón para procesar
        self.process_button = ttk.Button(
            self.parent, text="Procesar", command=self.process_data)
        self.process_button.grid(row=2, column=1, padx=5, pady=20)

        # Etiqueta de estado
        self.status_label = ttk.Label(self.parent, text="", foreground="green")
        self.status_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

    def select_zip(self):
        zip_path = filedialog.askopenfilename(
            filetypes=[("Archivo ZIP", "*.zip")])
        if zip_path:
            self.zip_entry.delete(0, tk.END)
            self.zip_entry.insert(0, zip_path)
            logger.info(f"Archivo ZIP seleccionado: {zip_path}")

    def select_output_folder(self):
        output_folder = filedialog.askdirectory()
        if output_folder:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, output_folder)
            logger.info(f"Carpeta de salida seleccionada: {output_folder}")

    def process_data(self):
        zip_path = self.zip_entry.get()
        output_folder = self.output_entry.get()

        if not os.path.isfile(zip_path):
            messagebox.showerror("Error", "Seleccione un archivo ZIP válido.")
            logger.error("Archivo ZIP inválido seleccionado.")
            return
        if not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Seleccione una carpeta de salida válida.")
            logger.error("Carpeta de salida inválida seleccionada.")
            return

        logger.info("Iniciando procesamiento de datos")
        self.status_label.config(text="Procesando...")
        self.process_button.state(['disabled'])
        threading.Thread(target=self.process_data_thread,
                         args=(zip_path, output_folder), daemon=True).start()

    def process_data_thread(self, zip_path, output_folder):
        try:
            # Definir rutas para Excel descomprimidos y archivo HDF5
            ruta_excel = os.path.join(output_folder, 'excel_descomprimidos')
            os.makedirs(ruta_excel, exist_ok=True)
            path_hdf5 = os.path.join(output_folder, 'data.h5')

            # Ejecutar el flujo principal en un proceso separado
            self.process = Process(target=flujo_principal, args=(
                zip_path,
                output_folder,
                ruta_excel,
                path_hdf5,
                os.path.join(output_folder, 'data_processing.log'),  # Ruta del archivo de log específico
                4  # Número de procesos en paralelo, ajusta según sea necesario
            ))
            self.process.start()
            logger.info("Proceso de procesamiento de datos iniciado.")

            # Esperar a que el proceso termine
            self.process.join()

            # Programar la actualización de la GUI en el hilo principal
            if self.process.exitcode == 0:
                self.parent.after(0, self.update_status, "Procesamiento completado con éxito.", "green")
                logger.info("Procesamiento de datos concluido con éxito")
            else:
                self.parent.after(0, self.update_status, "Error en el procesamiento. Revisa los logs.", "red")
                logger.error("El proceso de procesamiento de datos terminó con errores.")

        except Exception as e:
            self.parent.after(0, self.update_status, "", "green")
            messagebox.showerror("Error", f"Se produjo un error: {e}")
            logger.error(f"Error durante el procesamiento de datos: {e}")
        finally:
            self.parent.after(0, self.enable_process_button)

    def update_status(self, message, color):
        self.status_label.config(text=message, foreground=color)

    def enable_process_button(self):
        self.process_button.state(['!disabled'])
        logger.info("Proceso de datos finalizado")
