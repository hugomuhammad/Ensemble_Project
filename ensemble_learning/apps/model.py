import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder


def app():

    st.title('Model Page')

    #Upload file
    #menyimpan file yang diupload ke variabel dataset_name
    dataset_name = st.file_uploader("Choose a file")
    if dataset_name is not None:
        dataframe = pd.read_csv(dataset_name)    

    #menampilkan opsi model / classsifier
    #menyimpan pilihan pada variabel classifier_name
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Ensemble"))


    #fungsi praproses data
    def preprocess(dataset):
        cls = [] #list kolom dengan jumlah unique value = 2
        lst=[] #list kolom dengan tipe data object
        le = LabelEncoder() #fungsi transform categorical data -> numerical data
        dataframe = dataset

        #mencari kolom dengan unique value = 2
        for i in range (len(dataframe.columns)):         
            if dataframe[dataframe.columns[i]].nunique() == 2 or dataframe.columns[i] == 'Jalur Masuk':
                cls.append(i)
        #mengganti kolom dengan unique value = 2 dari data categorical -> numeric
        for i in cls:
            dataframe[dataframe.columns[i]] = le.fit_transform(dataframe[dataframe.columns[i]])
        
        #mencari data dengan tipe data object
        for i in range (len(dataframe.columns)):
            if dataframe[dataframe.columns[i]].dtype == "object":
                lst.append(i)

        #menghapus data dengan tipe data object
        dataframe = dataframe.drop(dataframe.columns[lst], axis=1)
        return dataframe




    #fungsi seleksi fitur
    def select_feature_target(dataset):
        #praproses data 
        dataframe = preprocess(dataset)
        #menentukan feature dan label
        X = dataframe.iloc[:,:-1]
        y = dataframe.iloc[:,-1:]
        return dataframe, X, y

    st.write(dataframe)

    #memanggil fungsi seleksi fitur
    data, X, y = select_feature_target(dataframe)
    
    st.markdown("## Feature dan Target")

    #menampilkan feature yang digunakan
    st.write('Feature',list(data.iloc[:,:-1].keys()))

    #menampilkan target
    st.write('Target',list(data.iloc[:,-1:].keys()))




    #fungsi menampilkan opsi input parameter
    def add_parameter_ui(clf_name):
        params = dict()

        #parameter SVM
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C

        #parameter KNN
        elif clf_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K

        #parameter ensemble learning
        else:
            K = st.sidebar.slider('Input K', 1, 15)
            params['K'] = K
            C = st.sidebar.slider('Input C', 0.01, 10.0)
            params['C'] = C
        return params

    #memanggil fungsi get params dan menyimpan ouutput pada variabel params
    params = add_parameter_ui(classifier_name)


    #fungsi menentukan model yang akan diapaki
    def get_classifier(clf_name, params):
        clf = None

        #SVM
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        
        #KNN
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        
        #ensemble KNN dan SVM
        else:
            knn = KNeighborsClassifier(n_neighbors=params['K'])
            svm = SVC(C=params['C'],kernel = 'poly', degree = 2 )
            clf = VotingClassifier( estimators= [('knn',knn),('svm',svm)], voting = 'hard')
        return clf

    #memanggil fungsi get_classifer dan menyimpan output pada variabel clf 
    clf = get_classifier(classifier_name, params)




    #fungsi hasil prediksi model
    def get_result(clf,clf_name,X,y):
        
        #assign variabel 
        y_pred = None
        acc = None
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        #menampilkan results untuk classifier SVM
        if clf_name == 'SVM':

            #train model
            clf.fit(X_train, y_train)
        
            #menampilkan pilihan data yang digunakan dari test data / upload data prediksi
            st.markdown("## Tujuan")
            display = ("Test Data", "Prediksi data")
            options = list(range(len(display)))
            value = st.selectbox("Pilih Tujuan", options, format_func=lambda x: display[x])
            df = False

            #apabila memilih test data
            if value == 0:
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
            
            #apabila memilih upload data prediksi 
            else:
                st.markdown('## Input Data Prediksi')
                input_file = st.file_uploader("choose a file")

                if input_file is not None:
                    dataframe = pd.read_csv(input_file) 
                    df = True  

                if df == True:   
                    st.write(dataframe)
                    dataframe = preprocess(dataframe)
                    X_test = dataframe.iloc[:, :-1]
                    y_pred = clf.predict(X_test)
                    acc = "Hasil Prediksi"

        #menampilkan results untuk classifier KNN
        elif clf_name == 'KNN':

            #train model
            clf.fit(X_train, y_train)
            
            #menampilkan pilihan data yang digunakan dari test data / upload data prediksi
            st.markdown("## Tujuan")
            display = ("Test Data", "Prediksi data")
            options = list(range(len(display)))
            value = st.selectbox("Pilih Tujuan", options, format_func=lambda x: display[x])
            df = False

            #apabila data yang dipilih adalah data test 
            if value == 0:
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
            
            #apabila memilih upload data prediksi 
            else:
                st.markdown('## Input Data Prediksi')
                input_file = st.file_uploader("choose a file")
                
                if input_file is not None:
                    dataframe = pd.read_csv(input_file) 
                    df = True  

                if df == True:   
                    st.write(dataframe)
                    dataframe = preprocess(dataframe)
                    X_test = dataframe.iloc[:, :-1]
                    y_pred = clf.predict(X_test)
                    acc = "Hasil Prediksi"
            
        #menampilkan results untuk classifier Ensemble learning
        else:

            #train model
            clf.fit(X_train, y_train)
            
            #menampilkan pilihan data yang digunakan dari test data / upload data prediksi
            st.markdown("## Tujuan")
            display = ("Test Data", "Prediksi data")
            options = list(range(len(display)))
            value = st.selectbox("Pilih Tujuan", options, format_func=lambda x: display[x])
            df = False

            #apabila data yang dipilih adalah data test 
            if value == 0:
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
            
            #apabila memilih upload data prediksi 
            else:
                st.markdown('## Input Data Prediksi')
                input_file = st.file_uploader("choose a file")
                
                if input_file is not None:
                    dataframe = pd.read_csv(input_file) 
                    df = True  

                if df == True:   
                    st.write(dataframe)
                    dataframe = preprocess(dataframe)
                    X_test = dataframe.iloc[:, :-1]
                    y_pred = clf.predict(X_test)
                    acc = "Hasil Prediksi"
            
        #return variabel predictor, label test data, hasil prediksi,dan akurasi
        return X_test, y_test ,y_pred, acc
            


    #KLASIFIKASI
    #memanggil fungsi get_result dan menyimpan output pada X_test, y_test, y_pred, acc
    X_test, y_test, y_pred, acc = get_result(clf, classifier_name, X, y)



    #MENAMPILKAN HASIL PREDIKSI
    st.markdown("## Hasil Prediksi")

    #menampilkan jenis model yang dipakai
    st.write(f'Classifier = {classifier_name}') 
    #menampilkan hasil akurasi
    st.write(f'Accuracy =', acc)

    #menampilkan dataframe
    st.write('Dataframe')
    X_test['Hasil_Prediksi'] = y_pred
    res =  X_test['Hasil_Prediksi'].map({0:'Diterima', 1:'Ditolak'})
    st.write(res)
