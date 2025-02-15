# custom_model_manager.py

import tkinter as tk
from tkinter import messagebox
import logging

logger = logging.getLogger('custom_model_manager')

class CustomModelManager:
    def __init__(self, parent, model_names, custom_model_params):
        self.parent = parent
        self.model_names = model_names
        self.custom_model_params = custom_model_params
        self.custom_model_window = None
        self.custom_params_entries = {}

    def add_custom_model(self):
        self.custom_model_window = tk.Toplevel(self.parent)
        self.custom_model_window.title("Definir Modelo Personalizado")
        self.custom_model_window.grab_set()  # Modal

        layers = ['Conv1', 'Conv2', 'Conv3', 'FC1']
        required_keys = ['Conv1_filters', 'Conv1_kernel', 'Conv2_filters', 'Conv2_kernel', 'Conv3_filters', 'Conv3_kernel', 'FC1_filters']

        for idx, layer in enumerate(layers):
            tk.Label(self.custom_model_window, text=f"{layer} Filters:").grid(row=idx, column=0, padx=5, pady=5, sticky='e')
            filters_entry = tk.Entry(self.custom_model_window, width=10)
            filters_entry.grid(row=idx, column=1, padx=5, pady=5, sticky='w')
            self.custom_params_entries[f'{layer}_filters'] = filters_entry

            if layer != 'FC1':
                tk.Label(self.custom_model_window, text=f"{layer} Kernel Size:").grid(row=idx, column=2, padx=5, pady=5, sticky='e')
                kernel_entry = tk.Entry(self.custom_model_window, width=10)
                kernel_entry.grid(row=idx, column=3, padx=5, pady=5, sticky='w')
                self.custom_params_entries[f'{layer}_kernel'] = kernel_entry

        save_button = tk.Button(self.custom_model_window, text="Guardar Modelo", command=self.save_custom_model)
        save_button.grid(row=len(layers), column=1, columnspan=2, pady=10)

    def save_custom_model(self):
        try:
            # Extraer parámetros ingresados por el usuario
            params = {}
            required_keys = ['Conv1_filters', 'Conv1_kernel', 'Conv2_filters', 'Conv2_kernel', 'Conv3_filters', 'Conv3_kernel', 'FC1_filters']
            for key in required_keys:
                entry = self.custom_params_entries.get(key)
                if entry:
                    value = entry.get()
                    if not value.isdigit():
                        raise ValueError(f"El valor para {key} debe ser un entero.")
                    params[key] = int(value)
                else:
                    raise ValueError(f"Falta el parámetro {key}.")

            # Crear un nombre único para el modelo personalizado
            existing_custom = [name for name in self.model_names if name.startswith('Modelo Personalizado')]
            model_count = len(existing_custom) + 1
            model_name = f"Modelo Personalizado {model_count}"
            self.model_names.append(model_name)

            # Almacenar los parámetros del modelo personalizado
            self.custom_model_params[model_name] = params

            logger.info(f"Modelo personalizado agregado: {model_name} con parámetros {params}")

            messagebox.showinfo("Éxito", f"{model_name} ha sido agregado exitosamente.")
            self.custom_model_window.destroy()
        except ValueError as ve:
            error_msg = f"Entrada inválida: {ve}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
        except Exception as e:
            error_msg = f"Se produjo un error: {e}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
