# metrics_display.py

import tkinter as tk
from tkinter import ttk

class MetricsDisplay:
    def __init__(self, parent):
        self.parent = parent
        self.metrics_tree = None
        self.setup_metrics_tree()

    def setup_metrics_tree(self):
        # Definir las columnas para la Treeview
        columns = ("Segmento", "Modelo", "Número de Datos", "MSE", "MAE", "RMSE")
        self.metrics_tree = ttk.Treeview(self.parent, columns=columns, show='headings')

        # Configurar las cabeceras y el ancho de las columnas
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, anchor='center', width=100)

        # Añadir una barra de desplazamiento vertical
        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        # Empaquetar la Treeview con expansión y relleno
        self.metrics_tree.pack(fill='both', expand=True, padx=5, pady=5)

        # Opcional: Aplicar estilos personalizados si es necesario
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
        style.configure("Treeview", font=('Helvetica', 10), rowheight=25)

    def display_metrics(self, df_metricas):
        # Limpiar la tabla existente
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        # Insertar nuevas métricas desde el DataFrame
        for _, row in df_metricas.iterrows():
            self.metrics_tree.insert("", tk.END, values=(
                row['Segmento'],
                row['Modelo'],
                row['Número de Datos'],
                f"{row['MSE']:.4f}",
                f"{row['MAE']:.4f}",
                f"{row['RMSE']:.4f}"
            ))
