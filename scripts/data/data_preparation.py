# data/data_preparation.py

import os
import torch
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, random_split
import logging
import gc
import concurrent.futures
from typing import Dict, Tuple, List, Optional

# Configuración del logger para el módulo
logger = logging.getLogger('data_preparation')
logger.setLevel(logging.DEBUG)

# Crear un manejador de archivos para registrar los logs
file_handler = logging.FileHandler('data_preparation.log')
file_handler.setLevel(logging.DEBUG)

# Crear un formato para los logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Añadir el manejador al logger si aún no tiene manejadores
if not logger.hasHandlers():
    logger.addHandler(file_handler)

def cargar_diccionario_desde_hdf5(path_archivo: str) -> Tuple[Dict[str, Dict[int, pd.DataFrame]], Dict[str, Dict[str, float]]]:
    """
    Carga los datos desde un archivo HDF5, los normaliza y los organiza en un diccionario.
    Calcula las estadísticas globales para futuras referencias (aunque ya no se usará para normalizar nuevamente).

    :param path_archivo: Ruta del archivo HDF5.
    :return: Diccionario con los datos normalizados y estadísticas globales.
    """
    dic_lista_dfs = {}
    try:
        with h5py.File(path_archivo, 'r') as hdf:
            for group in hdf.keys():
                group_dict = {}
                group_group = hdf[group]
                for cycle_index in group_group.keys():
                    data = group_group[cycle_index][()]
                    columns = list(group_group[cycle_index].attrs['columns'])
                    df = pd.DataFrame(data, columns=columns)
                    group_dict[int(cycle_index)] = df
                dic_lista_dfs[group] = group_dict
        logger.info(f"Datos cargados exitosamente desde '{path_archivo}'.")
    except FileNotFoundError:
        logger.error(f"Archivo HDF5 no encontrado: '{path_archivo}'.")
        raise
    except Exception as e:
        logger.error(f"Error al cargar datos desde '{path_archivo}': {e}")
        raise

    # Calcular estadísticas globales
    columnas = ['Voltage(V)', 'Current(A)', 'dV/dQ', 'Charge_Capacity(Ah)']
    estadisticas = calcular_estadisticas_globales(dic_lista_dfs, columnas)

    # Normalizar todos los DataFrames utilizando las estadísticas globales
    logger.info("Iniciando la normalización de todos los DataFrames cargados.")
    for group, cycles in dic_lista_dfs.items():
        for cycle, df in cycles.items():
            dic_lista_dfs[group][cycle] = normalizar_min_max(df, estadisticas, columnas)

    logger.info("Normalización completada para todos los DataFrames.")
    return dic_lista_dfs, estadisticas

def calcular_estadisticas_globales(dic_lista_dfs: Dict[str, Dict[int, pd.DataFrame]], columnas: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calcula los valores mínimos y máximos globales de las columnas especificadas.

    :param dic_lista_dfs: Diccionario de DataFrames.
    :param columnas: Lista de columnas a considerar.
    :return: Diccionario con estadísticas globales para cada columna.
    """
    concatenated_data = pd.concat(
        [df for group in dic_lista_dfs.values() for df in group.values()],
        ignore_index=True
    )

    estadisticas = {}
    for columna in columnas:
        col_data = concatenated_data[columna].dropna()
        estadisticas[columna] = {
            'min': col_data.min(),
            'max': col_data.max()
        }
    logger.info("Estadísticas globales calculadas para normalización.")
    return estadisticas

def normalizar_min_max(df: pd.DataFrame, estadisticas: Dict[str, Dict[str, float]], columnas: List[str]) -> pd.DataFrame:
    """
    Normaliza las columnas especificadas en el DataFrame usando Min-Max Scaling con estadísticas globales.

    :param df: DataFrame de entrada.
    :param estadisticas: Diccionario con estadísticas globales.
    :param columnas: Lista de columnas a normalizar.
    :return: DataFrame con las columnas normalizadas.
    """
    df_normalized = df.copy()
    for columna in columnas:
        if columna in df_normalized.columns:
            min_val = estadisticas[columna]['min']
            max_val = estadisticas[columna]['max']
            range_val = max_val - min_val
            if range_val != 0:
                df_normalized[columna] = (df_normalized[columna] - min_val) / range_val
                logger.debug(f"Columna '{columna}' normalizada usando Min-Max Scaling.")
            else:
                logger.warning(f"La columna '{columna}' tiene rango cero. No se normalizó.")
        else:
            logger.warning(f"Columna '{columna}' no encontrada en el DataFrame. Se omitió la normalización.")
    return df_normalized

def find_subset_by_number_of_data(
    df: pd.DataFrame, numero_de_datos: int = 7, part: int = 1
) -> Tuple[Optional[int], Optional[int]]:
    """
    Encuentra un subconjunto de datos con un número específico de muestras,
    permitiendo solapamiento entre partes basado en numero_de_datos.
    
    :param df: DataFrame de entrada.
    :param numero_de_datos: Número de datos requeridos en el subconjunto.
    :param part: Parte del DataFrame donde buscar (0 para todo el DataFrame, 1, 2, 3).
    :return: Índices de inicio y fin del subconjunto.
    """
    attempts = 0
    max_attempts = 100
    len_df = len(df)
    logger.debug(f"Buscando un subconjunto de {numero_de_datos} datos en la parte {part} del DataFrame con solapamiento.")

    while attempts < max_attempts:
        split_size = len_df // 3
        overlap = numero_de_datos - 1  # Tamaño del solapamiento

        if part == 0:
            # Usar todo el DataFrame
            start_range = 0
            end_range = len_df
        elif part == 1:
            start_range = 0
            end_range = split_size + overlap
        elif part == 2:
            start_range = split_size - overlap
            end_range = 2 * split_size + overlap
        elif part == 3:
            start_range = 2 * split_size - overlap
            end_range = len_df
        else:
            logger.error(f"Parte inválida: {part}. Debe ser 0, 1, 2 o 3.")
            return None, None

        # Aseguramos que los índices están dentro de los límites del DataFrame
        start_range = max(0, start_range)
        end_range = min(len_df, end_range)

        if end_range - start_range < numero_de_datos:
            logger.warning(f"No hay suficientes datos en la parte {part} del DataFrame.")
            return None, None

        start_idx = np.random.randint(start_range, end_range - numero_de_datos + 1)
        end_idx = start_idx + numero_de_datos

        if (end_idx - start_idx) == numero_de_datos:
            logger.debug(f"Subconjunto encontrado entre índices {start_idx} y {end_idx} después de {attempts} intentos.")
            return start_idx, end_idx
        attempts += 1

    logger.warning(f"No se encontró un subconjunto válido después de {max_attempts} intentos.")
    return None, None



def extract_subsets(lista_dfs: Dict[int, pd.DataFrame], estadisticas: Dict[str, Dict[str, float]],
                    numero_de_datos: int = 15, corte: int = 100, part: int = 2,
                    columnas: List[str] = ['Voltage(V)', 'Current(A)', 'dV/dQ']) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extrae subconjuntos de datos de los dataframes, asumiendo que ya están normalizados.

    :param lista_dfs: Diccionario de DataFrames de entrada.
    :param estadisticas: Diccionario con estadísticas globales (ya no se usa para normalizar).
    :param numero_de_datos: Número de datos a extraer en cada subconjunto.
    :param corte: Valor de corte para el ciclo (Cycle_Index).
    :param part: Parte del DataFrame donde buscar los datos.
    :param columnas: Lista de columnas a utilizar.
    :return: Lista de subconjuntos y valores finales.
    """
    subsets = []
    end_values = []
    logger.info(f"Iniciando extracción de subconjuntos con límite de {corte} y {numero_de_datos} datos por subset.")

    for key, df in lista_dfs.items():
        if key >= corte:
            logger.debug(f"El ciclo {key} es igual o mayor que el corte {corte}. Se omitirá.")
            continue

        if len(df) < numero_de_datos:
            logger.debug(f"El DataFrame del ciclo {key} tiene menos de {numero_de_datos} datos. Se omitirá.")
            continue

        # Asumir que el DataFrame ya está normalizado
        start_idx, end_idx = find_subset_by_number_of_data(df, numero_de_datos, part)
        if start_idx is not None and end_idx is not None:
            if end_idx < len(df):
                charge_capacity = df['Charge_Capacity(Ah)'].iloc[end_idx]
                if pd.notna(charge_capacity):
                    subset = df.loc[start_idx:end_idx, columnas].values
                    subsets.append(subset)
                    end_values.append(charge_capacity)
                    logger.debug(f"Subconjunto extraído del ciclo {key}, desde {start_idx} hasta {end_idx}.")
                else:
                    logger.warning(f"Charge_Capacity(Ah) es NaN en el índice {end_idx} del ciclo {key}. Se omitirá el subconjunto.")
            else:
                logger.warning(f"End index {end_idx} está fuera del rango del DataFrame para el ciclo {key}. Se omitirá el subconjunto.")
        else:
            logger.warning(f"No se pudo extraer un subconjunto válido para el ciclo {key}.")

    logger.info(f"Extracción completada. {len(subsets)} subconjuntos extraídos.")
    return subsets, end_values

def preparar_datos(subsets: List[np.ndarray], end_values: List[float],
                   batch_size: int = 64, train_size: float = 0.5,
                   val_size: float = 0.3, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepara los datos para entrenamiento, validación y prueba.

    :param subsets: Lista de subconjuntos de datos de entrada.
    :param end_values: Lista de valores finales.
    :param batch_size: Tamaño del batch para DataLoader.
    :param train_size: Proporción de datos para entrenamiento.
    :param val_size: Proporción de datos para validación.
    :param test_size: Proporción de datos para prueba.
    :return: DataLoaders para entrenamiento, validación y prueba.
    """
    logger.info("Preparando datos para entrenamiento, validación y prueba.")

    try:
        X = torch.tensor(subsets, dtype=torch.float32).permute(0, 2, 1)  # Cambiar dimensiones si es necesario
        Y = torch.tensor(end_values, dtype=torch.float32)

        dataset = TensorDataset(X, Y)
        total_size = len(dataset)
        train_size = int(train_size * total_size)
        val_size = int(val_size * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Datos preparados: {train_size} para entrenamiento, {val_size} para validación y {test_size} para prueba.")
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.error(f"Error al preparar los DataLoaders: {e}")
        raise
    finally:
        # Liberar memoria
        del X, Y, dataset, train_dataset, val_dataset, test_dataset
        gc.collect()

def process_group(args: Tuple[str, Dict[int, pd.DataFrame], int, int, int, Dict[str, Dict[str, float]], List[str]]) -> Tuple[List[np.ndarray], List[float]]:
    """
    Función auxiliar para procesamiento en paralelo.

    :param args: Tupla de argumentos (grupo_id, group_dict, numero_de_datos, corte, part, estadisticas, columnas).
    :return: Subconjuntos y valores normalizados.
    """
    grupo_id, group_dict, numero_de_datos, corte, part, estadisticas, columnas = args
    subsets = []
    end_values = []

    for cycle_index, df in group_dict.items():
        s, e_norm = extract_subsets({cycle_index: df}, estadisticas, numero_de_datos=numero_de_datos, corte=corte, part=part, columnas=columnas)
        subsets.extend(s)
        end_values.extend(e_norm)

    return subsets, end_values
