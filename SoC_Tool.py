# main.py
import tkinter as tk
from tkinter import ttk
import logging
from scripts.gui.process_tab import ProcessTab
from scripts.gui.visualize_tab import VisualizeTab
from scripts.gui.train_tab import TrainTab
from scripts.gui.estimate_tab import EstimateTab
#from scripts.gui.estimate_tab.estimate_tab import EstimateTab
#from scripts.gui.ecm_tab import ECMTab  # Asegúrate de que esta ruta es correcta

# Configurar el logger global
logging.basicConfig(
    level=logging.DEBUG,  # Cambiar a DEBUG para más detalles
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Aplicación iniciada")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento de Datos de Baterías")
        self.root.resizable(True, True)  # Permitir redimensionar la ventana
        self.custom_model_params = {}

        # Aplicar el tema moderno
        style = ttk.Style()
        style.theme_use('clam')

        # Crear las pestañas
        self.tab_control = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab4 = ttk.Frame(self.tab_control)
        self.tab5 = ttk.Frame(self.tab_control)
        self.tab6 = ttk.Frame(self.tab_control)  # Nueva pestaña

        self.tab_control.add(self.tab1, text='Procesar Datos')
        self.tab_control.add(self.tab2, text='Visualizar Datos')
        self.tab_control.add(self.tab3, text='Entrenar Modelos')
        self.tab_control.add(self.tab4, text='Estimación')
        #self.tab_control.add(self.tab5, text='Modelo ECM')  # Agregar la nueva pestaña

        self.tab_control.pack(expand=1, fill='both')

        # Inicializar y cargar cada pestaña
        self.process_tab = ProcessTab(self.tab1)
        self.visualize_tab = VisualizeTab(self.tab2)
        self.train_tab = TrainTab(self.tab3, self.custom_model_params)
        self.estimate_tab = EstimateTab(self.tab4, self.custom_model_params)
        #self.ecm_tab = ECMTab(self.tab5)  # Inicializar la nueva pestaña

        logger.info("Pestañas de la aplicación configuradas con éxito")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    logger.info("Aplicación finalizada")
