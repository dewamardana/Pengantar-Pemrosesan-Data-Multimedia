import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import graycomatrix, graycoprops

import pickle


st.title("Coba Klasifikasi")

# Directory untuk menyimpan file yang diunggah
upload_directory = "upload"
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))


# Fungsi untuk menyimpan file yang diunggah pengguna ke direktori tertentu
def save_input_data(input_data, upload_directory):
    try:
        if not os.path.exists(upload_directory):
            os.makedirs(upload_directory)
        save_path = os.path.join(upload_directory, input_data.name)
        with open(save_path, "wb") as f:
            f.write(input_data.getbuffer())
        return save_path
    except Exception as e:
        print(e)
        return None

def ekstraksi_fitur(img, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    props = ['correlation', 'homogeneity', 'contrast', 'energy']

    glcm = graycomatrix(img, distances=dists, angles=agls, levels=lvl, symmetric=sym, normed=norm)
    glcm_props = {name: graycoprops(glcm, name)[0] for name in props}

    angles = ['0', '45', '90','135']
    feature_dict = {}

    for name in props:
        for i, ang in enumerate(angles):
            key = f"{name}_{ang}"
            feature_dict[key] = [glcm_props[name][i]]

    return feature_dict

def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ymin, ymax, xmin, xmax = h//3, h*2//3, w//3, w*2//3
    crop = gray[ymin:ymax, xmin:xmax]
    resize = cv2.resize(crop, (0,0), fx=0.5, fy=0.5)
    return resize

def proses(input_data):
    if input_data is not None:
        save_path = save_input_data(input_data, upload_directory)
        if save_path:
            st.success("File berhasil diunggah dan disimpan")
            img = cv2.imread(save_path)

            if img is None:
                st.error("Maaf, gagal membaca gambar. Pastikan format file yang Anda unggah benar (jpg/jpeg/png).")
            else:
                st.image(img, channels="BGR")

                images = preprocessing(img)

            
                st.subheader("Hasil Setelah Preprocessing (Grayscale, Crop, Resize): ")
                st.image(images)

                # Ekstraksi fitur
                new_data_dict = ekstraksi_fitur(images)
                

                # Konversi ke DataFrame
                new_data_df = pd.DataFrame(new_data_dict)
                st.subheader("Fitur GLCM yang Diekstraksi:")
                st.dataframe(new_data_df.style.set_properties(**{'text-align': 'center'}))

                scaled_new_data = loaded_scaler.transform(new_data_df)

                # Lakukan prediksi
                prediction = loaded_model.predict(scaled_new_data)

                # Tampilkan hasil prediksi
                return prediction

        else:
            st.error("Gagal menyimpan file")
    else:
        st.warning("Silakan unggah gambar terlebih dahulu.")

def main():
    
    # Header dengan CSS
    st.markdown('<p class="main-header">Aplikasi Klasifikasi Kematangan Telur</p>', unsafe_allow_html=True)

    # User input untuk unggah file gambar
    input_data = st.file_uploader("Unggah Gambar Telur (Format: jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
    prediction = proses(input_data)
    if input_data is not None:
        st.write("Hasil Prediksi Klasifikasi:")
        if prediction == 1:
            st.write("Gambar Merupakan Telur Soft Boiled")
        elif prediction == 2:
            st.write("Gambar Merupakan Telur Medium Boiled")
        elif prediction == 3:
            st.write("Gambar Merupakan Telur Hard Boiled")
        else:
            st.write("Terjadi Kesalahan")
        
# Atur layout
st.markdown(
    """
    <style>
    .main-header {
        color: #008080;
        font-size: 24px;
        text-align: center;
    }
    .file-upload {
        padding: 20px;
        border: 2px dashed #008080;
        border-radius: 10px;
        text-align: center;
    }
    .file-upload:hover {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == '__main__':
    main()







