import streamlit as st
from multiapp import MultiApp
from apps import home, data, model # import your app modules here

app = MultiApp()

st.markdown("""
# Aplikasi Klasifikasi

Aplikasi Klasifikasi dengan Algoritma KNN, SVM, dan Ensemble.

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model", model.app)
# The main app
app.run()