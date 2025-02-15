# scripts/data/data_processing.py

import os
import zipfile
import tempfile
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import h5py
from datetime import datetime
import logging
import logging.handlers
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue, current_process, Process

# ================================
# Configuración del Logger
# ================================

def configure_logger(log_queue, log_file='data_processing.log'):
    """
    Configura el logger para manejar logs desde múltiples procesos.
    
    :param log_queue: Cola para manejar los logs.
    :param log_file: Archivo donde se guardarán los logs.
    :return: Instancia del logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Crear un manejador de cola
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    
    return logger

def logger_listener_process(log_queue, log_file='data_processing.log'):
    """
    Proceso que escucha la cola de logs y escribe en el archivo de log.
    
    :param log_queue: Cola desde donde se reciben los logs.
    :param log_file: Archivo donde se guardarán los logs.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Crear un manejador de archivo
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Crear un QueueListener que escuche la cola y maneje los logs
    listener = logging.handlers.QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()
    logger.debug("Logger Listener iniciado.")
    
    # Mantener el proceso activo hasta recibir un mensaje de parada
    while True:
        try:
            record = log_queue.get()
            if record is None:  # Señal para detener el listener
                break
            logger.handle(record)
        except Exception:
            import sys, traceback
            print('Error en logger_listener_process:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    
    listener.stop()
    logger.debug("Logger Listener detenido.")

# ================================
# Funciones de Procesamiento
# ================================

def mostrar_contenido_zip(zip_path):
    """
    Muestra el contenido de un archivo ZIP en formato de estructura de árbol.

    :param zip_path: Ruta del archivo ZIP.
    """
    logger = logging.getLogger()
    def print_tree(current_path, prefix=""):
        if current_path.is_dir():
            logger.info(prefix + "|-- " + current_path.name)
            prefix += "|   "
            for item in sorted(current_path.iterdir()):
                print_tree(item, prefix)
        else:
            logger.info(prefix + "`-- " + current_path.name)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            logger.info(f"Contenido del archivo ZIP '{zip_path}':")
            print_tree(Path(temp_dir))
    except zipfile.BadZipFile:
        logger.error(f"Archivo ZIP corrupto o inválido: {zip_path}")
    except Exception as e:
        logger.error(f"Error al mostrar contenido del ZIP '{zip_path}': {e}")

def descomprimir_zip_con_internos(zip_path, output_folder):
    """
    Descomprime un archivo ZIP y todos los archivos ZIP encontrados dentro,
    manteniendo la estructura original de subcarpetas.

    :param zip_path: Ruta del archivo ZIP.
    :param output_folder: Carpeta donde se extraerán los archivos.
    """
    logger = logging.getLogger()
    def descomprimir_archivo(zip_file_path, extract_to, zip_queue):
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                for file in zip_ref.namelist():
                    if file.endswith('.zip'):
                        full_path = os.path.join(extract_to, file)
                        internal_output_folder = os.path.join(extract_to, os.path.dirname(file))
                        os.makedirs(internal_output_folder, exist_ok=True)
                        zip_queue.append((full_path, internal_output_folder))
                logger.info(f"Descomprimido '{zip_file_path}' en '{extract_to}'")
        except zipfile.BadZipFile:
            logger.error(f"Archivo ZIP corrupto: {zip_file_path}")
        except Exception as e:
            logger.error(f"Error descomprimiendo '{zip_file_path}': {e}")

    zip_queue = [(zip_path, output_folder)]
    logger.info(f"Iniciando descompresión del archivo ZIP: {zip_path}")

    while zip_queue:
        current_zip, current_output_folder = zip_queue.pop(0)
        descomprimir_archivo(current_zip, current_output_folder, zip_queue)

def eliminar_archivos_zip_en_carpeta(carpeta):
    """
    Elimina todos los archivos ZIP en una carpeta y sus subcarpetas.

    :param carpeta: Ruta de la carpeta principal.
    """
    logger = logging.getLogger()
    try:
        archivos_zip = glob.glob(os.path.join(carpeta, '**', '*.zip'), recursive=True)
        for archivo_zip in archivos_zip:
            os.remove(archivo_zip)
            logger.info(f"Se eliminó '{archivo_zip}'")
    except Exception as e:
        logger.error(f"Error al eliminar archivos ZIP en '{carpeta}': {e}")

def manipular_dataframe(df):
    """
    Realiza varias manipulaciones en un DataFrame que contiene datos de ciclos de carga.

    :param df: DataFrame de entrada con los datos de los ciclos de carga.
    :return: DataFrame manipulado.
    """
    logger = logging.getLogger()
    try:
        logger.info("Iniciando manipulación de DataFrame.")
        # Crear una máscara para filtrar Step_Index entre 2 y 4
        mask = df['Step_Index'].between(2, 4)
        
        # Calcular la diferencia de Charge_Capacity(Ah)
        df['diff_Charge_Capacity(Ah)'] = df['Charge_Capacity(Ah)'].diff().fillna(0)
        
        # Inicializar Real_Charge_Capacity(Ah)
        df['Real_Charge_Capacity(Ah)'] = 0.0
        df.loc[mask, 'Real_Charge_Capacity(Ah)'] = df[mask].groupby('Cycle_Index')['diff_Charge_Capacity(Ah)'].cumsum()
        
        # Eliminar el primer ciclo si Cycle_Index == 1
        df = df[df['Cycle_Index'] != 1]
        
        # Filtrar Step_Index entre 1 y 5
        df_filtrado = df[(df['Step_Index'] >= 1) & (df['Step_Index'] <= 5)].copy()
        
        # Calcular dV, dQ y dV/dQ
        df_filtrado['dV'] = df_filtrado['Voltage(V)'].diff()
        df_filtrado['dQ'] = df_filtrado['Charge_Capacity(Ah)'].diff()
        df_filtrado['dV/dQ'] = df_filtrado.apply(
            lambda row: row['dV'] / row['dQ'] if row['dQ'] != 0 else np.nan, axis=1)
        
        # Eliminar Step_Index impares
        df_filtrado = df_filtrado[~df_filtrado['Step_Index'].isin([1, 3, 5])]
        
        # Seleccionar filas específicas dentro de cada grupo
        df_final = df_filtrado.groupby(['Cycle_Index', 'Step_Index'], group_keys=False).apply(
            lambda grupo: grupo.iloc[1:] if grupo.name[1] == 4 else grupo)
        
        # Actualizar Charge_Capacity(Ah) con Real_Charge_Capacity(Ah)
        df_final['Charge_Capacity(Ah)'] = df_final['Real_Charge_Capacity(Ah)']
        
        # Eliminar columnas temporales
        df_final.drop(columns=['Real_Charge_Capacity(Ah)', 'diff_Charge_Capacity(Ah)'], inplace=True)
        
        logger.info("Manipulación de DataFrame completada exitosamente.")
        return df_final
    except KeyError as ke:
        logger.error(f"Columna faltante en DataFrame: {ke}")
        raise
    except Exception as e:
        logger.error(f"Error al manipular DataFrame: {e}")
        raise

def procesar_excel_voltaje_corriente_dv_dq_modificado(excel_path):
    """
    Procesa un archivo Excel que contiene datos de voltaje y corriente para calcular dV/dQ.

    :param excel_path: Ruta del archivo Excel.
    :return: DataFrame manipulado.
    :raises ValueError: Si el archivo Excel no contiene más de una hoja.
    """
    logger = logging.getLogger()
    try:
        logger.info(f"Procesando archivo Excel: '{excel_path}'")
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        if len(sheet_names) > 1:
            df = pd.read_excel(excel_path, sheet_name=sheet_names[1])
            # Verificar si las columnas necesarias existen
            required_columns = ['Voltage(V)', 'Current(A)', 'Cycle_Index', 'Step_Index', 'Charge_Capacity(Ah)']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"El archivo Excel '{excel_path}' no contiene las columnas necesarias.")
                raise ValueError(f"El archivo Excel debe contener las columnas: {required_columns}")
            
            filtered_df = df[required_columns].copy()
            logger.info(f"Archivo Excel '{excel_path}' procesado con éxito.")
            return manipular_dataframe(filtered_df)
        else:
            logger.error("El archivo Excel debe contener más de una hoja.")
            raise ValueError("El archivo Excel debe contener más de una hoja.")
    except FileNotFoundError:
        logger.error(f"Archivo Excel no encontrado: '{excel_path}'")
        raise
    except Exception as e:
        logger.error(f"Error al procesar el archivo Excel '{excel_path}': {e}")
        raise

def encontrar_archivos_excel(ruta):
    """
    Encuentra todos los archivos Excel (.xlsx) en una carpeta y sus subcarpetas.

    :param ruta: Ruta de la carpeta principal.
    :return: Lista de rutas de archivos Excel encontrados.
    """
    logger = logging.getLogger()
    try:
        archivos_excel = []
        for raiz, _, archivos in os.walk(ruta):
            for archivo in archivos:
                if archivo.lower().endswith('.xlsx'):
                    archivos_excel.append(os.path.join(raiz, archivo))
        logger.info(f"Encontrados {len(archivos_excel)} archivos Excel en '{ruta}'.")
        return archivos_excel
    except Exception as e:
        logger.error(f"Error al buscar archivos Excel en '{ruta}': {e}")
        raise

def agrupar_y_ordenar_archivos(archivos):
    """
    Agrupa y ordena archivos por un identificador y una fecha en el nombre del archivo.

    :param archivos: Lista de rutas de archivos.
    :return: Diccionario con grupos de archivos ordenados por fecha.
    """
    logger = logging.getLogger()
    grupos = {}
    for archivo in archivos:
        nombre_archivo = os.path.basename(archivo)
        partes_nombre = nombre_archivo.split('_')
        if len(partes_nombre) < 5:
            logger.warning(f"Nombre de archivo inesperado, omitiendo: '{nombre_archivo}'")
            continue
        grupo = partes_nombre[1]
        fecha_str = "_".join(partes_nombre[2:5]).split('.')[0]
        try:
            fecha = datetime.strptime(fecha_str, "%m_%d_%y")
        except ValueError:
            logger.warning(f"Formato de fecha inválido en archivo: '{nombre_archivo}'")
            continue
        if grupo not in grupos:
            grupos[grupo] = []
        grupos[grupo].append((archivo, fecha))
    for grupo in grupos:
        grupos[grupo].sort(key=lambda x: x[1])
    logger.info("Archivos agrupados y ordenados por fecha exitosamente.")
    return grupos

def procesar_grupos_y_unir_parallel(archivos, max_workers=14):
    """
    Procesa grupos de archivos Excel en paralelo y los concatena en DataFrames por grupo.

    :param archivos: Lista de rutas de archivos Excel.
    :param max_workers: Número máximo de procesos en paralelo.
    :return: Diccionario con DataFrames concatenados por grupo.
    """
    logger = logging.getLogger()
    try:
        grupos = agrupar_y_ordenar_archivos(archivos)
        dfs_concatenados = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for grupo, archivos_fechas in grupos.items():
                futuros = []
                ultimo_cycle_index = 0
                for archivo, _ in archivos_fechas:
                    futuros.append(executor.submit(procesar_excel_voltaje_corriente_dv_dq_modificado, archivo))
                
                resultados = []
                for futuro in as_completed(futuros):
                    try:
                        df = futuro.result()
                        if ultimo_cycle_index > 0:
                            df['Cycle_Index'] += ultimo_cycle_index
                        ultimo_cycle_index = df['Cycle_Index'].max()
                        resultados.append(df)
                    except Exception as e:
                        logger.error(f"Error procesando archivo Excel: {e}")
                
                if resultados:
                    dfs_concatenados[grupo] = pd.concat(resultados, ignore_index=True)
                    logger.info(f"Datos concatenados para el grupo '{grupo}'.")
        return dfs_concatenados
    except Exception as e:
        logger.error(f"Error al procesar y unir grupos de archivos: {e}")
        raise

def separar_por_ciclos(dic_dfs):
    """
    Separa los datos de un DataFrame en múltiples DataFrames basados en el valor del Cycle_Index,
    creando un elemento de diccionario para cada ciclo.

    :param dic_dfs: Diccionario de DataFrames, donde cada clave es un grupo.
    :return: Nuevo diccionario con los datos separados por ciclos.
    """
    logger = logging.getLogger()
    dic_lista_dfs = {}
    try:
        for grupo, df in dic_dfs.items():
            dic_lista_dfs[grupo] = {}
            for cycle_index in df['Cycle_Index'].unique():
                df_filtrado = df[df['Cycle_Index'] == cycle_index][[
                    'Voltage(V)', 'Current(A)', 'dV/dQ', 'Charge_Capacity(Ah)']].reset_index(drop=True)
                dic_lista_dfs[grupo][cycle_index] = df_filtrado
        logger.info("Datos separados por ciclos exitosamente.")
        return dic_lista_dfs
    except Exception as e:
        logger.error(f"Error al separar datos por ciclos: {e}")
        raise

def guardar_diccionario_como_hdf5(dic_lista_dfs, path_archivo):
    """
    Guarda un diccionario de DataFrames en un archivo HDF5.

    :param dic_lista_dfs: Diccionario con los DataFrames a guardar.
    :param path_archivo: Ruta del archivo HDF5 a crear.
    """
    logger = logging.getLogger()
    try:
        with h5py.File(path_archivo, 'w') as hdf:
            for grupo, ciclos_dict in dic_lista_dfs.items():
                grupo_group = hdf.create_group(grupo)
                for cycle_index, df in ciclos_dict.items():
                    dataset = grupo_group.create_dataset(
                        str(cycle_index), data=df.to_numpy(), compression="gzip")
                    dataset.attrs['columns'] = df.columns.to_list()
        logger.info(f"Datos guardados exitosamente en el archivo HDF5: '{path_archivo}'.")
    except Exception as e:
        logger.error(f"Error al guardar datos en HDF5 '{path_archivo}': {e}")
        raise

def cargar_diccionario_desde_hdf5(path_archivo):
    """
    Carga un diccionario de DataFrames desde un archivo HDF5.

    :param path_archivo: Ruta del archivo HDF5.
    :return: Diccionario con los DataFrames cargados.
    """
    logger = logging.getLogger()
    dic_lista_dfs = {}
    try:
        with h5py.File(path_archivo, 'r') as hdf:
            for grupo in hdf.keys():
                grupo_dict = {}
                grupo_group = hdf[grupo]
                for cycle_index in grupo_group.keys():
                    data = grupo_group[cycle_index][()]
                    columns = grupo_group[cycle_index].attrs['columns']
                    df = pd.DataFrame(data, columns=columns)
                    grupo_dict[int(cycle_index)] = df
                dic_lista_dfs[grupo] = grupo_dict
        logger.info(f"Datos cargados exitosamente desde el archivo HDF5: '{path_archivo}'.")
    except FileNotFoundError:
        logger.error(f"Archivo HDF5 no encontrado: '{path_archivo}'")
        raise
    except Exception as e:
        logger.error(f"Error al cargar datos desde HDF5 '{path_archivo}': {e}")
        raise
    return dic_lista_dfs

# ================================
# Flujo Principal
# ================================

def flujo_principal(ruta_zip, carpeta_salida, ruta_excel, path_hdf5, log_file='data_processing.log', max_workers=14):
    """
    Flujo de trabajo principal para descomprimir, procesar y guardar datos.

    :param ruta_zip: Ruta al archivo ZIP principal.
    :param carpeta_salida: Carpeta donde se extraerán los archivos.
    :param ruta_excel: Ruta donde se encuentran los archivos Excel descomprimidos.
    :param path_hdf5: Ruta del archivo HDF5 de salida.
    :param log_file: Ruta del archivo de log.
    :param max_workers: Número máximo de procesos en paralelo.
    """
    # Crear una cola para los logs
    log_queue = Queue()
    
    # Iniciar el listener de logs en un proceso separado
    listener = Process(target=logger_listener_process, args=(log_queue, log_file))
    listener.start()
    
    # Configurar el logger en el proceso principal
    configure_logger(log_queue, log_file)
    
    logger = logging.getLogger()
    try:
        # Descomprimir archivos ZIP
        logger.info("Descomprimiendo archivos ZIP.")
        descomprimir_zip_con_internos(ruta_zip, carpeta_salida)
        
        # Eliminar archivos ZIP internos
        logger.info("Eliminando archivos ZIP internos.")
        eliminar_archivos_zip_en_carpeta(carpeta_salida)
        
        # Encontrar archivos Excel
        logger.info("Buscando archivos Excel.")
        archivos_excel = encontrar_archivos_excel(carpeta_salida)
        
        # Procesar y unir archivos Excel en paralelo
        logger.info("Procesando y uniendo archivos Excel.")
        dic_dfs = procesar_grupos_y_unir_parallel(archivos_excel, max_workers=max_workers)
        
        # Separar por ciclos
        logger.info("Separando datos por ciclos.")
        dic_lista_dfs = separar_por_ciclos(dic_dfs)
        
        # Guardar en HDF5
        logger.info("Guardando datos en archivo HDF5.")
        guardar_diccionario_como_hdf5(dic_lista_dfs, path_hdf5)
        
        logger.info("Flujo de procesamiento completado exitosamente.")
    except Exception as e:
        logger.error(f"Error en el flujo principal: {e}")
        raise
    finally:
        # Enviar señal para detener el listener
        log_queue.put_nowait(None)
        # Esperar a que el listener termine
        listener.join()
