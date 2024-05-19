import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import cv2
import os

import re
import sys
from skimage.feature import graycomatrix, graycoprops

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


st.title("Modeling")

df = pd.read_csv('GLCM_Telur.csv')

# Membuat DataFrame untuk visualisasi df.shape
shape_data = pd.DataFrame({
    'Dimensi': ['Rows', 'Columns'],
    'Jumlah': [df.shape[0], df.shape[1]]
})

# Menampilkan DataFrame shape_data
st.write("Data Shape DataFrame:")
st.write(shape_data)

# Menambahkan garis pemisah
st.markdown("---")

st.header("Cleaning Data")
st.subheader("Apa itu Data Null?")

# Menambahkan keterangan tentang data null
st.write("""
         
<div style="text-align:justify">
Data null atau missing data adalah data yang tidak tersedia atau hilang dalam dataset. 
Hal ini dapat terjadi karena berbagai alasan, seperti kesalahan dalam proses pengumpulan data, 
entri data yang tidak lengkap, atau karena data tersebut memang tidak ada.

Dalam konteks pandas DataFrame, nilai null biasanya direpresentasikan dengan `NaN` (Not a Number).
</div>
""", unsafe_allow_html=True)

# Menampilkan jumlah data null di setiap kolom
st.write("Jumlah Data Null di Setiap Kolom:")
st.dataframe(df.isnull().sum())

# Menambahkan garis pemisah
st.markdown("---")

st.subheader("Data Duplikat")
st.write("""

Data duplikat merujuk pada baris-baris dalam dataset yang memiliki nilai yang sama untuk semua kolomnya. 

Mengapa Data Duplikat Perlu Dihapus?
<div style="text-align:justify">
1. **Pengaruh pada Analisis Statistik:*Data duplikat dapat mempengaruhi analisis statistik dengan meningkatkan bobot atau kepentingan dari beberapa nilai yang sama, menghasilkan hasil yang bias.
2. **Waktu Komputasi:** Memproses data duplikat dapat memperlambat waktu komputasi, terutama dalam analisis besar.
3. **Ketepatan Model:** Data duplikat dapat memberikan informasi yang tidak berguna atau menyebabkan overfitting pada model.

Oleh karena itu, penting untuk mengidentifikasi dan menghapus data duplikat agar analisis data dapat dilakukan dengan akurat dan efisien.
</div>
""", unsafe_allow_html=True)
duplikasi = df.duplicated().sum()
df.drop_duplicates(inplace=True)

st.write("Jumlah Duplikat Dalam Data Dataset: " + str(duplikasi))

st.markdown("---")

st.header("Histogram Distribusi Dari Setiap Kolom ")
st.write("""
Histogram adalah sebuah jenis grafik yang digunakan untuk merepresentasikan distribusi data numerik. 
Grafik ini terdiri dari batang-batang yang masing-masing mewakili kelompok (atau interval) dari nilai-nilai data. 
Tinggi setiap batang menunjukkan frekuensi atau jumlah kejadian nilai-nilai data dalam kelompok tersebut.
         """)

columns_to_plot = [
    'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
    'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135',
    'contrast_0', 'contrast_45', 'contrast_90', 'contrast_135',
    'energy_0', 'energy_45', 'energy_90', 'energy_135'
]

# Tentukan ukuran grid
n_rows = 4
n_cols = 4

# Buat figure dan axis
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))

# Flatten axis untuk memudahkan iterasi
axes = axes.flatten()

# Iterasi melalui setiap kolom dan buat plot
for i, col in enumerate(columns_to_plot):
    sns.histplot(data=df, x=col, bins=10, kde=True, color='#91008a', ax=axes[i])
    axes[i].set_title(f'Distribusi {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frekuensi')

# Mengatur layout agar lebih rapi
plt.tight_layout()

# Menampilkan plot di Streamlit
st.pyplot(fig)

st.markdown("---")

st.header("Presentase Jumlah Label")
st.write("""
Dengan menggunakan diagram Batang ini, kita dapat dengan mudah melihat distribusi kematangan telur rebus dalam dataset secara visual, 
memungkinkan kita untuk mengidentifikasi label mana yang paling umum atau dominan, serta proporsi relatif dari masing-masing label.
         """)

# Membuat plot persebaran label
st.write("Persebaran Label:")
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', palette='rocket')
plt.title('Presentasi Kematangan Telur Rebus')
plt.xlabel('Hasil / Label')
plt.ylabel('Frekuensi')

# Menampilkan plot di Streamlit
st.pyplot(plt)

st.markdown("---")

st.header("Korelasi Antar Data")
st.write("""
Korelasi adalah sebuah ukuran statistik yang digunakan untuk mengukur hubungan antara dua variabel. Dalam konteks analisis data, korelasi menggambarkan seberapa erat hubungan atau ketergantungan antara dua variabel. Korelasi tidak menyiratkan sebab dan akibat, tetapi hanya mengukur sejauh mana dua variabel bergerak bersama-sama.

Terdapat beberapa jenis korelasi yang umum digunakan, yang paling populer adalah korelasi Pearson. Korelasi Pearson mengukur sejauh mana dua variabel bergerak bersama-sama dalam arah yang sama (korelasi positif) atau arah yang berlawanan (korelasi negatif). Korelasi Pearson memiliki rentang nilai antara -1 hingga 1, di mana:

Korelasi 1 menunjukkan hubungan linier positif sempurna antara variabel.
Korelasi -1 menunjukkan hubungan linier negatif sempurna antara variabel.
Korelasi 0 menunjukkan tidak adanya hubungan linier antara variabel.
Korelasi yang lebih dekat ke 1 atau -1 menunjukkan hubungan yang lebih kuat antara variabel, sedangkan nilai yang lebih dekat ke 0 menunjukkan hubungan yang lebih lemah.
         """)
plt.figure(figsize=(6, 4))

# Membuat heatmap dari korelasi DataFrame
heatmap = sns.heatmap(df.corr(), cmap='BuPu')

# Menampilkan heatmap di Streamlit
st.pyplot(heatmap.figure)


X = df.drop(columns=['label'])
Y = df['label']
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# Membuat dan melatih model
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

# Memprediksi target
y_pred = clf.predict(x_test)

st.markdown("---")

st.header("Confusion Matriks")
st.write("""
Confusion matrix adalah alat evaluasi kritis dalam klasifikasi yang memungkinkan kita untuk memahami kinerja model secara detail. 
Ini adalah tabel yang menggambarkan jumlah prediksi yang benar dan salah yang dibuat oleh model, dibagi menjadi empat bagian: True Positive (TP), True Negative (TN), False Positive (FP), dan False Negative (FN). 
True Positive adalah jumlah kasus di mana model dengan benar memprediksi kelas positif, sementara True Negative adalah jumlah kasus di mana model dengan benar memprediksi kelas negatif. 
False Positive adalah jumlah kasus di mana model salah memprediksi kelas positif, sementara False Negative adalah jumlah kasus di mana model salah memprediksi kelas negatif. 
Dengan menggunakan confusion matrix, kita dapat menghitung berbagai metrik evaluasi klasifikasi, seperti precision, recall, dan f1-score,
 """)


# Membuat confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Menampilkan confusion matrix di Streamlit
st.write("Confusion Matrix:")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='seismic', fmt='g',
            xticklabels=['Label 1', 'Label 2', 'Label 3'],
            yticklabels=['Label 1', 'Label 2', 'Label 3'])
plt.xlabel('Prediksi')
plt.title('Confusion Matrix')
st.pyplot(fig)


st.write("""
### Deskripsi Metrik Evaluasi Klasifikasi

**Precision:** Precision adalah proporsi dari prediksi positif yang benar dibandingkan dengan total prediksi positif yang dilakukan oleh model. Dinyatakan sebagai:
""")

st.write(r"Precision: $Precision = \frac{True Positives}{True Positives + False Positives}$")
st.write("""
Precision mengukur seberapa akurat model dalam mengklasifikasikan data positif. Sebuah nilai precision yang tinggi menunjukkan bahwa model memiliki sedikit false positives.

**Recall (Sensitivity):** Recall adalah proporsi dari data positif yang diprediksi dengan benar dibandingkan dengan total data positif yang seharusnya diprediksi oleh model. Dinyatakan sebagai:
""")
st.write(r"Recall: $Recall = \frac{True Positives}{True Positives + False Negatives}$")
st.write("""
Recall mengukur seberapa baik model dalam menangkap data positif. Sebuah nilai recall yang tinggi menunjukkan bahwa model memiliki sedikit false negatives.

**F1-Score:** F1-score adalah rata-rata harmonik dari precision dan recall. Dinyatakan sebagai:
""")
st.write(r"F1-Score: $F1-Score =  \frac{2 \times Precision \times Recall}{Precision + Recall}$")
st.write("""
F1-score mengukur keseimbangan antara precision dan recall. Ini adalah metrik yang berguna ketika Anda ingin mencari model yang memiliki trade-off yang baik antara precision dan recall.

**Support:** Support adalah jumlah kemunculan aktual dari setiap kelas dalam data pengujian. Dalam laporan klasifikasi, ini adalah jumlah sampel yang sesuai dengan setiap kelas target.

Dengan menggunakan keempat metrik ini, Anda dapat mendapatkan pemahaman yang lebih baik tentang kinerja model klasifikasi Anda dalam mengklasifikasikan data. Sebagai contoh, Anda mungkin ingin model yang memiliki nilai precision, recall, dan f1-score yang tinggi, sambil memperhatikan jumlah support untuk masing-masing kelas.
""")


# Menghitung akurasi
CLF_acc = accuracy_score(y_pred, y_test)

# Mendapatkan laporan klasifikasi
report = classification_report(y_test, y_pred)

# Menampilkan laporan klasifikasi dan akurasi di Streamlit
st.header("Laporan Klasifikasi:")
st.text(report)
st.text("Akurasi SVM: {:.2f}%".format(CLF_acc * 100))

