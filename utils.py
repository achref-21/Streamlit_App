import streamlit as st
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


@st.cache_data
def process_file(data):
    scaler = StandardScaler()
    scaler.fit_transform(data[["ORDERED_QUANTITY"]])
    # Step 1: AutoARIMA to find the best fit ARIMA model parameters
    auto_arima_model = auto_arima(data["ORDERED_QUANTITY"], start_p=2, start_q=2, max_p=4, max_q=4, d=1, seasonal=True,
                                  m=7)
    order = auto_arima_model.order
    seasonal_order = auto_arima_model.seasonal_order

    # Step 2: Fit ARIMA model
    arima_model = ARIMA(data["ORDERED_QUANTITY"], order=order, seasonal_order=seasonal_order)
    arima_result = arima_model.fit()

    # Step 3: Extracting data
    fitted_values = arima_result.fittedvalues
    residuals = arima_result.resid

    # Step 4: Seasonal Decomposition of Residualss
    decomposition = seasonal_decompose(residuals, model='arima', period=7)
    seasonal, trend, resid = decomposition.seasonal, decomposition.trend, decomposition.resid
    data = data.drop(["SUPPLIER_NUMBER"], axis=1)
    data = data.drop(["FOLIO_NUMBER"], axis=1)
    data["fitted_values"] = fitted_values
    data["trend_values"] = trend
    data["seasonal_values"] = seasonal
    data["resid_values"] = resid
    return data, arima_result, scaler


def create_sequences(data):
    scaler = StandardScaler()
    column_names = data.columns
    index = data.index
    data_scaled = scaler.fit_transform(data)
    final_data_scaled = pd.DataFrame(data_scaled, columns=column_names, index=index)
    # Assuming 'final_train_data' is your DataFrame and it's already loaded.
    # Extracting necessary columns as features
    features = final_data_scaled[
        ['ORDERED_QUANTITY', 'fitted_values', 'trend_values', 'seasonal_values', 'resid_values']]

    # Convert DataFrame to numpy array for processing
    data_array = features.to_numpy()

    # Define the sequence length
    sequence_length = 7

    # Initialize lists to hold the sequences and their corresponding targets
    input_sequences = []
    target_data = []

    # Create sequences using a sliding window
    for i in range(len(data_array) - sequence_length):
        # Extract the sequence and the next value
        seq = data_array[i:i + sequence_length]  # Sequence of 7 rows
        target = data_array[i + sequence_length]  # Next row (change as needed for your specific target setup)

        # Append the sequence and target to their respective lists
        input_sequences.append(seq)
        target_data.append(
            target[0])

    # Convert lists to numpy arrays for model input
    input_sequences = np.array(input_sequences)
    input_sequences = np.nan_to_num(input_sequences, nan=0.0)
    target_data = np.array(target_data)
    return input_sequences


@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    absolute_error = tf.abs(error)
    quadratic_part = tf.minimum(absolute_error, delta)
    linear_part = absolute_error - quadratic_part
    return 0.5 * tf.square(quadratic_part) + delta * linear_part


def denormalize(scaler, predictions):
    inversed_predictions = scaler.inverse_transform(predictions)
    return inversed_predictions
