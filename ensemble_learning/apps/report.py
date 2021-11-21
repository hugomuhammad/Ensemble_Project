from pandas.core.frame import DataFrame
from apps import model as md
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def app():

    #fungsi visualisasi data
    def countPlot(column, data):
        fig = plt.figure(figsize=(10, 4))
        plt.title('Kategori UKT calon penerima beasiswa')
        sns.countplot(x = column, data = data)
        st.pyplot(fig)

    #Mencetak dataframe
    st.write('Penerima Beasiswa')
    report_df = md.results
    st.write(report_df)

    #Mencetak grafik
    st.write('Report')
    countPlot('ukt', report_df)

    #fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Use the axes for plotting
    #sns.countplot(x='ukt', data=report_df, ax=axes[0])
    #axes[0].set_xlabel('UKT')
    #axes[0].set_ylabel('Jumlah Penerima UKT')
    #axes[0].set_title('UKT Calon Penerima Beasiswa')

    # Use the axes for plotting
    #sns.countplot(x='Jalur masuk', data=report_df, ax=axes[1])
    #axes[1].set_xlabel('Jalur masuk')
    #axes[1].set_ylabel('Jumlah')
    #axes[1].set_title('Jalur masuk Calon Penerima Beasiswa')

