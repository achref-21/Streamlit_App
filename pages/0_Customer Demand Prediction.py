import streamlit as st
import pandas as pd
from utils import process_file, custom_loss, create_sequences, denormalize
from keras.initializers import Orthogonal
import pickle as pck
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

st.title("Upload your file for preprocessing")
supplier = st.selectbox("Supplier", ["Pick supplier", "Supplier1", "Supplier3", "Supplier5", "Supplier6", "Supplier7", "Supplier8", "Supplier9", "Supplier12"])
input = st.file_uploader(label="Upload here")
data = pd.DataFrame()

model_dict = {
    'Supplier1': 'models/model_1.h5',
    'Supplier3': 'models/model(Huber)_supp_3.h5',
    'Supplier5': 'models/model(Huber)_supp_5.h5',
    'Supplier6': 'models/model(Huber)_supp_6.h5',
    'Supplier7': 'models/model(Huber)_supp_7.h5',
    'Supplier8': 'models/model(Huber)_supp_8.h5',
    'Supplier9': 'models/model(Huber)_supp_9.h5',
    'Supplier12': 'models/model(Huber)_supp_12.h5'
}

custom_objects = {
    'Orthogonal': Orthogonal
}


def load_and_compile_model(supplier):
    model_path = model_dict.get(supplier)
    optimizer = Adam(learning_rate=0.0001)
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    model.compile(optimizer=optimizer, loss=custom_loss)
    return model


if input:
    df = pd.read_excel(input)
    df = df.set_index(df["FOLIO_NUMBER"])
tab1, tab2, tab3 = st.tabs(["Raw data", "Transformed data", "Prediction"])
with tab1:
    if input:
        chart = st.line_chart(df["ORDERED_QUANTITY"])


with tab2:
    if input:
        data, arima_model, scaler = process_file(df)
    if not data.empty:
        chart2 = st.line_chart(data)
with tab3:
    model = load_and_compile_model(supplier)
    st.write(model)
    if not data.empty:
        sequences = create_sequences(data)
        predictions = model.predict(sequences)
        inversed_predictions = denormalize(scaler, predictions)
        chart3 = st.line_chart(inversed_predictions)
