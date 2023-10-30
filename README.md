# Laporan Proyek Machine Learning
### Nama :Ariya Wibawa Pratama
### Nim :211351025
### Kelas :Pagi A

## Domain Proyek

proyek ini dapat digunakan untuk  memprediksi apakah tingkat keparahan kanker paru-paru yang diderita pasien tinggi, sedang atau rendah dengan menggunakan metode algoritma Decission Tree. Kanker paru-paru adalah penyebab utama kematian akibat kanker di seluruh dunia, terhitung 1,59 juta kematian pada tahun 2018. 

## Business Understanding

Untuk memprediksi tingkat keparahan kanker paru-paru


### Problem Statements

- Dalam kasus ini, masalah yang saya jelajahi adalah masalah klasifikasi.

saya akan mencaari informasi dan memprediksi tingkat keparahan kanker paru paru pada seseorang

### Goals

menemukan klasifikasi dan inforamsi penyakit kanker paru-paru seseorang tinggi,sedang atau lemah.

### Solution statements
- Membangun suatu sistem yang dapat mempelajari suatu data (Machine Learning) memprediksi tingkat keparahan kanker paru-paru
- Sistem berjalan dengan menggunakan metode klasifikasi yang dinilai cocok untuk memprediksi
## Data Understanding
Dataset yang digunakan berasal dari situs Kaggle. Dataset ini mengandung 999 entries dan 24 columns<br><br>

[Lung Cancer Prediction](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link) .

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada kanker paru-paru adalah sebagai berikut:


   - Patient_Id = Number Identitas Pasien Peyakit Kanker Paru-paru.

   - Age = Umur Pasien Kanker Paru-paru.

   - Gender = Jenis kelamin Pasien Kanker Paru-paru.

   - Air Pollution = Paparan polusi udara yang mengganggu pernafasan pasien.

   - Alcohol use = Tingkat penggunaan alkohol yang di konsumsi pasien.

   - Dust Allergy = Apakah Pasien mengalami alergi debu.

   - OccuPational Hazards = Bahaya pekerjaan yang dialami oleh pasien di tempat kerjanya. Jenis bahaya pekerjaan bisa berupa bahaya bahan kimia, bahaya bilogis, bahaya        			    psikososial dan bahaya fisik.

   - Genetic Risk = Risiko genetik yang diwariskan dari orang tua pasien melalui gen.

   - chronic Lung Disease = Penyakit paru-paru kronis yang di derita pasien ini dapat mengakibatkan sulit untuk bernafas, dikarenakan paru-paru mengalami peradangan 			    dalam jangka waktu yang lama.

   - Balanced Diet = Diet seimbang atau diet sehat pasien ini untuk membantu, menjaga dan meningkatkan kesehatan pasien secara keseluruhan. Diet ini menyediakan gizi 		     yang penting bagi tubuh, yaitu cairan, makronutrein, mikronutren, dan energi makanan yang cukup.

   - Obesity = Suatu gangguan yang melibatkan lemak tubuh berlebihan yang meningkat risiko masalah kesehatan Pasien. Obesitas sering kali terjadi karena kalori yang 	       masuk lebih banyak, daripada yang di bakar melalui olahraga dan kegiatan sehari-hari.

   - Smoking = Riwayat merokok pada pasien.

   - Passive Smoker = Pasien yang mempunyai riwayat perokok pasif atau seseorang yang menghirup asap rokok dari perokok aktif. Pasien yang mempunyai riwayat ini lebih 		      berbahaya menyebabkan penyakit paru-paru.

   - Chest Pain = Gejala nyeri dada pada pasien dalam waktu singkat yang dapat menyebabkan kematian.

   - Coughing of Blood = Batuk berdarah yang di alami oleh pasien dari Paru-paru.

   - Fatigue = Perubahan dari keadaan pasien dari yang lebih kuat mejadi keadaan yang lebih lemah.

   - Weight Loss = Penurunan berat baadan secara keseluruhan yang di alami oleh pasien.

   - Shortness of Breath = Masalah kesehatan yang membuat seseorang kesulitan untuk menghirup udara.

   - Wheezing = Mengi, suara khas yang berasal dari saluran pernapasan yang menyempit. Mengi ini menghasilkan siulan yang akan terdengar jelas ketika penderita 		menghirup dan menghembuskan nafas.

   - Swallowing Difficulty = Kesulitan menelan yang di alami pasien karena memasukkan makanan terlalu banyak makanan ke dalam mulut, tidak mengunyah makanan dengan 			     benar, mulut kering, pil, atau makanan terlalu panas.

   - Clubbing of Finger Nails = Kondisi jari-jari tangan atau kaki pasien yang membengkak. Kondisi ini terjadi akibat kelainan genetik, atau gangguan di paru-paru dan 			        jantung.

   - Frequent Cold = kondisi frekuensi dingin pasien.

   - Dry Cough = Salah satu jenis batuk yang tidak mengeluarkan lendir atau dahak, batuk ini biasanya terjadi akibat udara sekitar pasien kering.

   - Snoring = Suara mendengkur pasien saat tidur. Terjadi saat rasa lelah setelah melakukan aktivitas banyak dan berat sehari-hari.

   - level = Label yang membahas tingkat keparahan kanker paru-paru pada pasien :

        - High = 0
        - Medium = 2
        - Low = 1



## Data Preparation
pertama tama kita import libary yang di butuhkan
```bash
import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
```
Setelah itu kita akan men-definsikan dataset menggunakan fungsi pada library pandas
```bash
df = pd.read_csv("cancer_patient.csv")
```
melihat beberapa baris pertama dari sebuah DataFrame.
```bash
df.head()
```
Lalu kita akan melihat informasi mengenai dataset dengan syntax seperti dibawah:
```bash
df.info()
```
```bash
independen = [col for col in df.columns != 'Level']
defenden = 'Level'
```
memisahkan numerik dan kategori
```bash
# memisahkan numerik dan kategori
numerical = []
catgcols = []

for col in df.columns:
    if df[col].dtype == 'float64':
        numerical.append(col)
    elif df[col].dtype == 'int64':
        numerical.append(col)
    else:
        catgcols.append(col)

for col in df.columns:
    if col in numerical:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)
```
check numerical
```bash
numerical
```
check category cols
```bash
catgcols
```
check label
```bash
df['Level'].value_counts()
```
```bash
df['Patient_Id'].value_counts()
```
masukan libary 
```bash
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in catgcols:
    df[col] = le.fit_transform(df[col])
```
proses tranformasi
```bash
df['Level'] = le.fit_transform(df['Level'])
```
menampilkan 5 baris teratas 
```bash
df.head()
```
data yang sudah clean 
```bash
df.to_csv('LungCancer.csv')
```
```bash
df['Level'].value_counts()
```
```bash
df['Patient_Id'].value_counts()
```
```bash
X = df.drop(columns=['Level', 'index', 'Frequent_Cold', 'Snoring'], axis=1)
Y = df['Level']
```
```bash
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```
```bash
print(X.shape, x_train.shape, x_test.shape)
```
(1000, 22) (800, 22) (200, 22)

## Modeling
membuat model decision tree
```bash
desicion = DecisionTreeClassifier(
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0.0,
    random_state=42, splitter='best'
)
```
membuat desicion fit
```bash
model = desicion.fit(x_train, y_train)
```
```bash
akurasi data training 
x_train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict, y_train)
```
```bash
print('Akurasi data training : ', training_data_accuracy)
```
hasil Akurasi data training :  0.99125

akurasi data testing
```bash
x_test_predict = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict, y_test)```
```bash
print('Akurasi data testing : ', test_data_accuracy)
```
hasil Akurasi data testing :  0.985
## Evaluation
```bash
plt.figure(figsize = (11, 9))
plt.title("Lung Cancer Chances Due to Air Polution")
plt.pie(df['Level'].value_counts(), explode = (0.1, 0.02, 0.02), labels = ['High', 'Medium', 'Low'], autopct = "%1.2f%%", shadow = True)
plt.legend(title = "Lung Cancer Chances", loc = "lower left")
```
```bash
fig = plt.figure(figsize=(25, 25))
_ = tree.plot_tree(
    model,
    feature_names = independen,
    class_names = ['Low', 'Medium', 'High'],
    filled = True)
```


## Deployment
```bash
import pickle

filename = 'LungCancer.sav'
pickle.dump(model, open(filename, 'wb'))
```


