# models/model_training.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import logging
import gc
import pandas as pd
import h5py
import numpy as np
import concurrent.futures

from scripts.data.data_preparation import (
    cargar_diccionario_desde_hdf5,
    extract_subsets,
    preparar_datos,
    process_group,
    calcular_estadisticas_globales  # Importar la función
)
# Configuración del logger
logger = logging.getLogger('model_training')

# Detectar la disponibilidad de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Dispositivo de entrenamiento: {device}")

if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    logger.info("CUDNN Benchmark activado para optimización de rendimiento.")

def train_model_with_validation(model, train_loader, val_loader, epochs=150, learning_rate=0.001, progress_callback=None, device=torch.device('cpu')):
    """
    Entrena el modelo con validación y retorna las pérdidas de entrenamiento y validación.

    :return: Lista de pérdidas de entrenamiento y validación por época.
    """
    logger.info(f"Iniciando el entrenamiento del modelo con {epochs} épocas y una tasa de aprendizaje de {learning_rate}.")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)  # Mover inputs al dispositivo
            labels = labels.to(device)  # Mover labels al dispositivo
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_train_loss = total_loss / len(train_loader)
        train_losses.append(average_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        val_losses.append(average_val_loss)

        logger.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}')

        # Reportar progreso si el callback está definido
        if progress_callback:
            progress_callback(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")

    logger.info("Entrenamiento completado.")
    return train_losses, val_losses

def evaluate_model(model, dataloader, device=torch.device('cpu')):
    """
    Evalúa el modelo y calcula métricas.

    :param model: El modelo a evaluar.
    :param dataloader: DataLoader con los datos de prueba.
    :param device: Dispositivo para evaluación (CPU o GPU).
    :return: MSE, MAE, RMSE, todas las predicciones, todas las etiquetas.
    """
    logger.info("Iniciando evaluación del modelo.")
    all_predictions = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)  # Mover inputs al dispositivo
            labels = labels.to(device)  # Mover labels al dispositivo
            predictions = model(inputs).squeeze().cpu().numpy()  # Mover predicciones a CPU
            labels = labels.cpu().numpy()  # Mover labels a CPU
            all_predictions.extend(predictions)
            all_labels.extend(labels)

    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)

    logger.info(f"Evaluación completada: MSE = {mse:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
    return mse, mae, rmse, all_predictions, all_labels

# Definición de una clase base para los modelos
class BaseCNNModel(nn.Module):
    def __init__(self, input_channels, conv_filters, conv_kernels, fc_units):
        super(BaseCNNModel, self).__init__()
        layers = []
        in_channels = input_channels

        # Construir capas convolucionales
        for out_channels, kernel_size in zip(conv_filters, conv_kernels):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.Tanh())
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(in_features=in_channels, out_features=fc_units)
        self.bn_fc = nn.BatchNorm1d(fc_units)
        self.act_fc = nn.Tanh()
        self.fc_out = nn.Linear(in_features=fc_units, out_features=1)
        self.act_out = nn.ELU()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.act_fc(self.bn_fc(self.fc1(x)))
        x = self.act_out(self.fc_out(x))
        return x

# Función para crear modelos específicos
def create_model(model_name, input_channels, custom_params=None, device=None):
    """
    Crea una instancia del modelo especificado.

    :param model_name: Nombre del modelo (e.g., 'Model 1', 'Model 2', 'Modelo Personalizado 1').
    :param input_channels: Número de canales de entrada.
    :param custom_params: Parámetros personalizados para modelos personalizados.
    :param device: Dispositivo para el modelo (CPU o GPU).
    :return: Instancia del modelo.
    """
    if model_name == 'Model 1':
        # Parámetros para Model 1
        conv_filters = [32, 64, 128]
        conv_kernels = [3, 3, 3]
        fc_units = 256
    elif model_name == 'Model 2':
        # Parámetros para Model 2
        conv_filters = [44, 42, 54]
        conv_kernels = [7, 4, 16]
        fc_units = 60
    elif model_name == 'Model 3':
        # Parámetros para Model 3
        conv_filters = [11, 60, 56]
        conv_kernels = [10, 1, 14]
        fc_units = 60
    elif model_name.startswith('Modelo Personalizado') and custom_params:
        # Parámetros personalizados
        conv_filters = [
            custom_params['Conv1_filters'],
            custom_params['Conv2_filters'],
            custom_params['Conv3_filters']
        ]
        conv_kernels = [
            custom_params['Conv1_kernel'],
            custom_params['Conv2_kernel'],
            custom_params['Conv3_kernel']
        ]
        fc_units = custom_params['FC1_filters']
    else:
        raise ValueError(f"Modelo {model_name} no reconocido o faltan parámetros.")

    model = BaseCNNModel(input_channels, conv_filters, conv_kernels, fc_units)
    if device is not None:
        model = model.to(device)
    return model

def entrenar_evaluar_modelos(dic_lista_dfs_and_stats, modelos_seleccionados, segmentos_seleccionados, parametros, save_folder, custom_model_params, progress_callback=None):
    """
    Función principal para entrenar y evaluar los modelos.
    """
    logger.info("Iniciando el proceso de entrenamiento y evaluación de modelos.")
    dic_lista_dfs, estadisticas = dic_lista_dfs_and_stats
    metricas = []
    n_iterations = parametros['n_iterations']
    corte = parametros['corte']
    epochs = parametros['epochs']
    fecha_actual = datetime.now().strftime('%Y_%m_%d')

    numero_de_datos_list = parametros['numero_de_datos']
    if not isinstance(numero_de_datos_list, list):
        numero_de_datos_list = [numero_de_datos_list]

    segment_map = {'A': 1, 'B': 2, 'C': 3, 'F': 0}

    # Crear una carpeta para los datos de prueba
    test_data_folder = os.path.join(save_folder, 'test_data')
    os.makedirs(test_data_folder, exist_ok=True)

    for numero_de_datos in numero_de_datos_list:
        for segment in segmentos_seleccionados:
            logger.info(f"Iniciando segmento {segment} con número de datos {numero_de_datos}.")
            subsets = []
            end_values = []
            part = segment_map.get(segment, 1)

            # Preparar los argumentos para las tareas
            tasks = []
            for i in range(n_iterations):
                for group_id, group_dict in dic_lista_dfs.items():
                    tasks.append((group_id, group_dict, numero_de_datos, corte, part, estadisticas, ['Voltage(V)', 'Current(A)', 'dV/dQ']))

            # Ejecutar las tareas en paralelo
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                results = executor.map(process_group, tasks)

                # Recopilar los resultados
                for result in results:
                    s, e_norm = result
                    if s is not None and e_norm is not None:
                        subsets.extend(s)
                        end_values.extend(e_norm)

            logger.info(f"Segmento {segment} con número de datos {numero_de_datos} completado. Subconjuntos extraídos: {len(subsets)}")

            # Verificar si hay datos suficientes
            if not subsets:
                msg = f"No hay datos suficientes para el segmento {segment} con número de datos {numero_de_datos}."
                logger.warning(msg)
                if progress_callback:
                    progress_callback(msg)
                continue

            # Preparar los datos
            try:
                train_loader, val_loader, test_loader = preparar_datos(subsets, end_values)
                logger.info(f"Datos preparados para el segmento {segment} con número de datos {numero_de_datos}.")
            except Exception as e:
                logger.error(f"Error al preparar datos para el segmento {segment} con número de datos {numero_de_datos}: {e}")
                continue

            # Entrenar y evaluar cada modelo con los datos del segmento actual
            for model_name in modelos_seleccionados:
                logger.info(f"Iniciando entrenamiento para {model_name} en segmento {segment} con número de datos {numero_de_datos}.")

                # Crear una instancia del modelo y moverlo al dispositivo
                try:
                    if model_name.startswith('Modelo Personalizado'):
                        params = custom_model_params.get(model_name)
                    else:
                        params = None
                    model = create_model(model_name, input_channels=3, custom_params=params, device=device)
                    msg = f"Entrenando {model_name} en segmento {segment} con número de datos {numero_de_datos}."
                    logger.info(msg)
                    if progress_callback:
                        progress_callback(msg)
                except Exception as e:
                    msg = f"Error al instanciar el modelo {model_name} en segmento {segment} con número de datos {numero_de_datos}: {e}"
                    logger.error(msg)
                    if progress_callback:
                        progress_callback(msg)
                    continue

                # Entrenar el modelo
                try:
                    train_losses, val_losses = train_model_with_validation(
                        model, train_loader, val_loader, epochs=epochs, learning_rate=0.001,
                        progress_callback=progress_callback, device=device
                    )
                    logger.info(f"Entrenamiento completado para {model_name} en segmento {segment} con número de datos {numero_de_datos}.")
                except Exception as e:
                    msg = f"Error durante el entrenamiento del modelo {model_name} en segmento {segment} con número de datos {numero_de_datos}: {e}"
                    logger.error(msg)
                    if progress_callback:
                        progress_callback(msg)
                    continue

                # Guardar las pérdidas de entrenamiento
                try:
                    losses_df = pd.DataFrame({
                        'Epoch': list(range(1, epochs + 1)),
                        'Train Loss': train_losses,
                        'Val Loss': val_losses
                    })
                    losses_filename = f'{model_name}_Segment_{segment}_Data{numero_de_datos}_losses.csv'
                    losses_save_path = os.path.join(save_folder, losses_filename)
                    losses_df.to_csv(losses_save_path, index=False)
                    msg = f"Pérdidas de entrenamiento guardadas en {losses_save_path}"
                    logger.info(msg)
                    if progress_callback:
                        progress_callback(msg)
                except Exception as e:
                    msg = f"Error al guardar las pérdidas de entrenamiento para el modelo {model_name} en segmento {segment} con número de datos {numero_de_datos}: {e}"
                    logger.error(msg)
                    if progress_callback:
                        progress_callback(msg)
                    continue

                # Evaluar el modelo
                try:
                    mse, mae, rmse, _, _ = evaluate_model(model, test_loader, device=device)
                    logger.info(f"Evaluación completada para {model_name} en segmento {segment} con número de datos {numero_de_datos} - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                except Exception as e:
                    msg = f"Error durante la evaluación del modelo {model_name} en segmento {segment} con número de datos {numero_de_datos}: {e}"
                    logger.error(msg)
                    if progress_callback:
                        progress_callback(msg)
                    continue

                # Guardar el modelo entrenado
                try:
                    model_filename = f'{model_name}_Segment_{segment}_Data{numero_de_datos}.pth'
                    model_save_path = os.path.join(save_folder, model_filename)
                    torch.save(model.state_dict(), model_save_path)
                    msg = f"Modelo guardado como {model_save_path}"
                    logger.info(msg)
                    if progress_callback:
                        progress_callback(msg)
                except Exception as e:
                    msg = f"Error al guardar el modelo {model_name} en segmento {segment} con número de datos {numero_de_datos}: {e}"
                    logger.error(msg)
                    if progress_callback:
                        progress_callback(msg)
                    continue

                # Guardar los datos de prueba
                try:
                    test_data = []
                    test_labels = []
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        test_data.append(inputs.cpu().numpy())
                        test_labels.append(labels.cpu().numpy())

                    test_data = np.concatenate(test_data, axis=0)
                    test_labels = np.concatenate(test_labels, axis=0)

                    test_data_filename = f'{model_name}_Segment_{segment}_Data{numero_de_datos}_test.h5'
                    test_data_path = os.path.join(test_data_folder, test_data_filename)
                    with h5py.File(test_data_path, 'w') as hdf:
                        hdf.create_dataset('test_data', data=test_data)
                        hdf.create_dataset('test_labels', data=test_labels)
                    msg = f"Datos de prueba guardados como {test_data_path}"
                    logger.info(msg)
                    if progress_callback:
                        progress_callback(msg)
                except Exception as e:
                    msg = f"Error al guardar los datos de prueba para el modelo {model_name} en segmento {segment} con número de datos {numero_de_datos}: {e}"
                    logger.error(msg)
                    if progress_callback:
                        progress_callback(msg)
                    continue

                # Agregar métricas
                metricas.append({
                    'Número de Datos': numero_de_datos,
                    'Segmento': segment,
                    'Modelo': model_name,
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse
                })
                logger.info(f"Finalizado el modelo {model_name} para el segmento {segment} con número de datos {numero_de_datos}.")

                # Liberar memoria después de procesar cada modelo
                del model
                gc.collect()
                logger.debug("Memoria liberada después de procesar el modelo.")

    df_metricas = pd.DataFrame(metricas)
    logger.info("\nResumen de Métricas:")
    logger.info(df_metricas)
    print("\nResumen de Métricas:")
    print(df_metricas)

    # Guardar las métricas en un archivo CSV
    metrics_save_path = os.path.join(save_folder, f'metricas_{fecha_actual}.csv')
    try:
        df_metricas.to_csv(metrics_save_path, index=False)
        msg = f"Métricas guardadas en {metrics_save_path}"
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    except Exception as e:
        msg = f"Error al guardar las métricas en {metrics_save_path}: {e}"
        logger.error(msg)
        if progress_callback:
            progress_callback(msg)

    logger.info("Proceso de entrenamiento y evaluación de modelos completado.")
    return df_metricas
