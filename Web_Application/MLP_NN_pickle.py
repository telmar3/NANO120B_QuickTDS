import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

st.set_page_config(
     page_title="TDS parameter predictions",
     page_icon="house",
     layout="wide",
     menu_items={
         'Get Help': 'mailto:mher@ucsd.edu',
         'Report a bug': 'mailto: mher@ucsd.edu@ucsd.edu',
         'About': "# Made for NanoEnginering Capston Project."
     }
 )

start_time = time.time()

import streamlit as st
import pandas as pd
import pickle

def load_model(url):
    return pickle.load(open(url,'rb'))

model = load_model('MLP_NN_simple.pkl')

# Define the input interface
st.write("Input the following process conditions to make predictions:")
with st.form("data_input"):
    st.write("Upload data as a .csv or .dat file:")
    data_input = st.file_uploader("Choose a file", type=["csv", "dat"])
    submit_button = st.form_submit_button(label='Submit')

# Load and preprocess the data
if submit_button:
    data = pd.read_csv(data_input, sep='\t') if data_input.name.endswith('.dat') else pd.read_csv(data_input)
    
    if data.empty:
        st.write("Error: data file is empty.")
    elif data.shape[1] == 1:
        st.write("This is a csv of the desportion flux.")
        # Downsample to 301 rows if the length is greater than 301
        if len(data) > 301:
            skip_factor = len(data) // 301
            selected_data_column_2 = data[::skip_factor][:301]
        else:
            selected_data_column_2 = data
        # Create a list of indices to select every nth row
        index_list = list(range(0, len(selected_data_column_2)))
        st.write(len(index_list))
        st.write(selected_data_column_2.shape)
        st.write(selected_data_column_2.index)

        # Create a new dataframe with the downsampled second column
        data = pd.DataFrame(selected_data_column_2.iloc[index_list, 0].values, index=index_list, columns=['des_flux'])

        ## Add missing rows if necessary
        if data.shape[0] < model.n_features_in_:
            missing_rows = np.zeros((model.n_features_in_ - data.shape[0], 1))
            data = np.concatenate((data.values, missing_rows), axis=0)


        # Make predictions and display results
        predictions = model.predict(data)
        st.write("Predicted values:")
        st.write(predictions)

    else:
        st.write("This is a dat file. Make sure the second column is of the desportion flux.")
        data = data.iloc[:, 1:] # Use only the second column

        # Downsample to 301 rows if the length is greater than 301
        if len(data) > 301:
            skip_factor = len(data) // 301
            selected_data_column_2 = data[::skip_factor][:301]
        else:
            selected_data_column_2 = data.iloc[:, 1]
        # Create a list of indices to select every nth row
        index_list = list(range(0, len(selected_data_column_2)))
        st.write(len(index_list))
        st.write(selected_data_column_2.shape)
        st.write(selected_data_column_2.index)

        # Create a new dataframe with the downsampled second column
        data = pd.DataFrame(selected_data_column_2.iloc[index_list, 0].values, index=index_list, columns=['des_flux'])

        ## Add missing rows if necessary
        if data.shape[0] < model.n_features_in_:
            missing_rows = np.zeros((model.n_features_in_ - data.shape[0], 1))
            data = np.concatenate((data.values, missing_rows), axis=0)


        # Make predictions and display results
        predictions = model.predict(data)
        st.write("Predicted values:")
        st.write(predictions)

