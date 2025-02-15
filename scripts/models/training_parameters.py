# training_parameters.py

import logging
from tkinter import messagebox

logger = logging.getLogger('training_parameters')

def get_training_parameters(n_iter_entry, n_data_values, corte_entry, epochs_entry):
    try:
        n_iterations = int(n_iter_entry.get())
        corte = int(corte_entry.get())
        epochs = int(epochs_entry.get())
        # Asegurarse de que n_data_values es una lista de enteros
        if not isinstance(n_data_values, list) or not all(isinstance(n, int) for n in n_data_values):
            raise ValueError("Los números de datos deben ser una lista de enteros.")
        parametros = {
            'n_iterations': n_iterations,
            'numero_de_datos': n_data_values,
            'corte': corte,
            'epochs': epochs
        }
        return parametros
    except ValueError as e:
        error_msg = f"Ingrese valores numéricos válidos en los parámetros: {e}"
        logger.error(error_msg)
        messagebox.showerror("Error", error_msg)
        return None
