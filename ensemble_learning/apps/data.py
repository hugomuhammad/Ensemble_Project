import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    st.title('Data Page')
    
    st.write('Simple Classification App')
    st.write('Input data untuk training model')

    df = False
    dataset_name = st.file_uploader("Choose a file")
    if dataset_name is not None:
        dataframe = pd.read_csv(dataset_name)
        df = True      
        
  

    st.write("Dataframe")
    if df == True:
        st.write(dataframe)

