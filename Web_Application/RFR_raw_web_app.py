import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="TDS RFR raw predictions",
    page_icon="house",
    layout="wide",
    menu_items={
        'Get Help': 'mailto:mher@ucsd.edu',
        'Report a bug': 'mailto: mher@ucsd.edu@ucsd.edu',
        'About': "Made for the University of San Diego's NanoEngineering Capstone Project Spring 2023."
    }
)

# Loading the trained model
@st.cache_data
def load_model(url):
    return pickle.load(open(url, 'rb'))

model = load_model('rfr_raw.pkl')

# Displaying the Time
def convertMillis(millisecs):
    seconds = int((millisecs / 1000) % 60)
    minutes = int((millisecs / (1000 * 60)) % 60)
    hours = int((millisecs / (1000 * 60 * 60)) % 24)

    return f"{hours}:{minutes}:{seconds}"


# Define the input interface
st.write("Input the following process conditions to make predictions:")
with st.form("data_input"):
    st.write("Upload data as a .dat file:")
    data_input = st.file_uploader("Choose a file", type=["dat"])
    submit_button = st.form_submit_button(label='Submit')

# Load and preprocess the data
if submit_button:
    start_time = time.time()

    data = pd.read_csv(data_input, sep='\t') if data_input.name.endswith('.dat') else pd.read_csv(data_input)

    if data.empty:
        st.write("Error: data file is empty.")

    else:
        st.write("This is a dat of the desportion flux. Make sure the second column is of the desportion flux.")
        data = data.iloc[:, 1:]  # Use only the second column

        # Downsample to 301 rows if the length is greater than 301
        if len(data) > 301:
            skip_factor = len(data) // 301
            selected_data_column_2 = data[::skip_factor][:301]
        else:
            selected_data_column_2 = data.iloc[:, 1]
        # Create a list of indices to select every nth row
        index_list = list(range(0, len(selected_data_column_2)))

        # Create a new dataframe with the downsampled second column
        data = pd.DataFrame(selected_data_column_2.iloc[index_list, 0].values, index=index_list,
                            columns=['des_flux'])

        ## Add missing rows if necessary
        if data.shape[0] < model.n_features_in_:
            missing_rows = np.zeros((model.n_features_in_ - data.shape[0], 1))
            data = np.concatenate((data.values, missing_rows), axis=0)

        # Rescaling
        des_flux = data['des_flux']
        des_flux = des_flux.tolist()
        num_files = int(len(des_flux) / 301)
        des_flux = np.reshape(des_flux, [num_files, 301])

        # Scaling the data according to the model
        scaler = StandardScaler(with_mean=True, with_std=True)
        des_flux_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

        # Make predictions and display results
        predictions = model.predict(des_flux_scaled.reshape(-1, 301))

        # Separate the predictions for detrap_en and defect_con
        detrap_en_predictions = 10 ** predictions[:, :4].reshape(-1)
        defect_con_predictions = 10 ** predictions[:, 4:].reshape(-1)


        # Create a DataFrame to store the predicted values
        pred_param = pd.DataFrame({
            'detrap_en': detrap_en_predictions,
            'defect_con': defect_con_predictions
        })

        st.write("Predicted values:")
        st.write(pred_param)
        st.write("Shape of des_flux_scaled:", des_flux_scaled.shape)
        st.write("Shape of predictions:", predictions.shape)

        # Load the training dataset and extract target values
        training_data = pd.read_csv('training_data_1.csv', header=None) 
        # Extract the second and third columns
        second_column = training_data.iloc[:, 1]
        third_column = training_data.iloc[:, 2]

        # Get the first four rows of the second column
        detrap_en = second_column.head(4)
        def_conc = third_column.head(4)

        # Compare the predicted values with the first 4 training labels
        for i in range(len(detrap_en)):
            predicted_value_detrap_en = pred_param['detrap_en'].values[i]  # Get the predicted value for detrap_en
            predicted_value_def_conc = pred_param['defect_con'].values[i]  # Get the predicted value for defect_conc
            label_detrap_en = detrap_en[i]  # Get the corresponding training label for detrap_en
            label_def_conc = def_conc[i]  # Get the corresponding training label for defect_conc
            difference_detrap_en = predicted_value_detrap_en - label_detrap_en
            difference_def_conc = predicted_value_def_conc - label_def_conc

            st.write("Parameters", i+1)
            st.write("Predicted detrap_en:", predicted_value_detrap_en)
            st.write("Simulation detrap_en:", label_detrap_en)
            st.write("Difference detrap_en:", difference_detrap_en)

            st.write("Predicted defect_conc:", predicted_value_def_conc)
            st.write("Simulation defect_conc:", label_def_conc)
            st.write("Difference defect_conc:", difference_def_conc)


    elapsed_time = time.time() - start_time
    st.write("Elapsed Time: " + convertMillis(elapsed_time))
